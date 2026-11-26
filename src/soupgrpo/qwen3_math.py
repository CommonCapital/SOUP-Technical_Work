"""Qwen3 decoder-layer math, written as plain functions over a weight dict.

The engine owns where weights live (pinned CPU store -> GPU ring buffer); this
module only computes. Matches HF transformers Qwen3 numerics:
  - RMSNorm in fp32, cast back, then multiply by weight
  - per-head q_norm / k_norm before RoPE
  - RoPE cos/sin computed in fp32, cast to compute dtype
  - GQA attention via SDPA, scale head_dim**-0.5
  - SwiGLU MLP

LoRA: y = x W^T + scale * ((x A^T) B^T), A: [r, in], B: [out, r], computed in
fp32 and cast back. W is frozen; A and B are trainable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_HAS_GQA = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5)

PROJ_SHAPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Qwen3Dims:
    hidden: int
    intermediate: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    rope_theta: float
    rms_eps: float
    tie_embeddings: bool

    @classmethod
    def from_hf_config(cls, c) -> "Qwen3Dims":
        get = c.get if isinstance(c, dict) else (lambda k, d=None: getattr(c, k, d))
        hidden = get("hidden_size")
        n_heads = get("num_attention_heads")
        rope_theta = get("rope_theta")
        if rope_theta is None:  # newer configs nest it under rope_parameters
            rp = get("rope_parameters") or {}
            rope_theta = rp.get("rope_theta", 1_000_000.0)
        return cls(
            hidden=hidden, intermediate=get("intermediate_size"), n_layers=get("num_hidden_layers"),
            n_heads=n_heads, n_kv_heads=get("num_key_value_heads"),
            head_dim=get("head_dim") or hidden // n_heads, vocab=get("vocab_size"),
            rope_theta=float(rope_theta), rms_eps=get("rms_norm_eps", 1e-6),
            tie_embeddings=bool(get("tie_word_embeddings", False)),
        )

    def proj_io(self) -> dict[str, tuple[int, int]]:
        """(in_features, out_features) per projection."""
        h, i, qd, kvd = self.hidden, self.intermediate, self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        return {"q_proj": (h, qd), "k_proj": (h, kvd), "v_proj": (h, kvd), "o_proj": (qd, h),
                "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h)}


# HF tensor name suffix -> short name used in the engine
LAYER_TENSORS = {
    "input_layernorm.weight": "ln1",
    "post_attention_layernorm.weight": "ln2",
    "self_attn.q_proj.weight": "q_proj",
    "self_attn.k_proj.weight": "k_proj",
    "self_attn.v_proj.weight": "v_proj",
    "self_attn.o_proj.weight": "o_proj",
    "self_attn.q_norm.weight": "q_norm",
    "self_attn.k_norm.weight": "k_norm",
    "mlp.gate_proj.weight": "gate_proj",
    "mlp.up_proj.weight": "up_proj",
    "mlp.down_proj.weight": "down_proj",
}


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """position_ids [B, T] -> cos, sin [B, T, head_dim]."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.int64).float() / head_dim))
    freqs = position_ids.float()[..., None] * inv  # [B, T, d/2]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x [B, H, T, d]; cos/sin [B, T, d]
    return x * cos[:, None] + _rotate_half(x) * sin[:, None]


def linear(x, w, lora: dict | None, name: str, scale: float):
    y = F.linear(x, w)
    if lora is not None and name in lora:
        a, b = lora[name]
        y = y + (scale * F.linear(F.linear(x.float(), a), b)).to(y.dtype)
    return y


@dataclass
class KVSlot:
    """Pre-allocated KV cache for one layer: [B, n_kv, T_max, d]."""
    k: torch.Tensor
    v: torch.Tensor


def decoder_layer(x, w: dict, d: Qwen3Dims, cos, sin, attn_mask, lora: dict | None = None,
                  lora_scale: float = 1.0, kv: KVSlot | None = None, kv_pos: int | None = None):
    """One Qwen3 decoder layer.

    x: [B, T, H]. attn_mask: bool [B, 1, T, T_k] (True = attend).
    kv/kv_pos: if given, this call's K/V are written into kv at
    [kv_pos, kv_pos+T) and attention runs over kv[:, :, :kv_pos+T].
    """
    B, T, _ = x.shape
    cd = w["q_proj"].dtype  # compute dtype; x may be fp32 (fp16 mode keeps the residual in fp32)
    h = rms_norm(x, w["ln1"], d.rms_eps).to(cd)
    q = linear(h, w["q_proj"], lora, "q_proj", lora_scale).view(B, T, d.n_heads, d.head_dim)
    k = linear(h, w["k_proj"], lora, "k_proj", lora_scale).view(B, T, d.n_kv_heads, d.head_dim)
    v = linear(h, w["v_proj"], lora, "v_proj", lora_scale).view(B, T, d.n_kv_heads, d.head_dim)
    q = rms_norm(q, w["q_norm"], d.rms_eps).transpose(1, 2)
    k = rms_norm(k, w["k_norm"], d.rms_eps).transpose(1, 2)
    v = v.transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if kv is not None and isinstance(kv_pos, tuple):
        # per-row write positions (speculative decoding: rows accept different numbers of tokens).
        # kv_pos = (start [B] long tensor, kv_len int): row b writes slots start[b]..start[b]+T-1,
        # attention reads slots [0, kv_len) under the caller's mask.
        start, kv_len = kv_pos
        idx = start[:, None] + torch.arange(T, device=start.device)[None]        # [B, T]
        rows = torch.arange(B, device=start.device)[:, None]
        kv.k[rows, :, idx] = k.transpose(1, 2)
        kv.v[rows, :, idx] = v.transpose(1, 2)
        k, v = kv.k[:, :, :kv_len], kv.v[:, :, :kv_len]
    elif kv is not None:
        kv.k[:, :, kv_pos:kv_pos + T] = k
        kv.v[:, :, kv_pos:kv_pos + T] = v
        k, v = kv.k[:, :, :kv_pos + T], kv.v[:, :, :kv_pos + T]
    if d.n_heads != d.n_kv_heads and not _HAS_GQA:  # torch < 2.5
        rep = d.n_heads // d.n_kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=d.head_dim ** -0.5)
    else:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=d.head_dim ** -0.5,
                                           enable_gqa=d.n_heads != d.n_kv_heads)
    o = o.transpose(1, 2).reshape(B, T, d.n_heads * d.head_dim)
    x = x + linear(o, w["o_proj"], lora, "o_proj", lora_scale)
    h = rms_norm(x, w["ln2"], d.rms_eps).to(cd)
    g = linear(h, w["gate_proj"], lora, "gate_proj", lora_scale)
    u = linear(h, w["up_proj"], lora, "up_proj", lora_scale)
    return x + linear(F.silu(g) * u, w["down_proj"], lora, "down_proj", lora_scale)


def build_attn_mask(attention_mask: torch.Tensor, q_len: int, k_len: int) -> torch.Tensor:
    """attention_mask [B, k_len] (1 = real token). Query i sits at key index
    k_len - q_len + i. Returns bool [B, 1, q_len, k_len]. Padded query rows
    attend to themselves only, so softmax never sees an all-masked row (their
    outputs are ignored)."""
    dev = attention_mask.device
    qi = torch.arange(k_len - q_len, k_len, device=dev)[:, None]
    ki = torch.arange(k_len, device=dev)[None, :]
    causal = ki <= qi                                        # [q, k]
    m = causal[None] & attention_mask[:, None, :].bool()     # [B, q, k]
    self_only = (ki == qi)[None].expand_as(m)
    q_is_pad = ~attention_mask[:, k_len - q_len:].bool()     # [B, q]
    m = torch.where(q_is_pad[:, :, None], self_only, m)
    return m[:, None]


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions count only real tokens, so left or right padding gives the
    same positions for the same text."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
