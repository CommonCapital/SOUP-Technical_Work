"""Tiny random Qwen3 models for testing the engine on a laptop CPU.

Same architecture as Qwen3-4B (q/k norm, GQA, tied embeddings, SwiGLU,
RoPE theta 1e6), tiny sizes. Weights get a larger init std than HF's 0.02 so
logits are far from uniform and errors are visible.
"""

from __future__ import annotations

from pathlib import Path

import torch


def tiny_config(n_layers: int = 4, vocab: int = 512, hidden: int = 64, heads: int = 4, kv_heads: int = 2,
                head_dim: int = 16, intermediate: int = 128):
    from transformers import Qwen3Config
    return Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=n_layers,
        num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim,
        max_position_embeddings=4096, rope_theta=1_000_000.0, rms_norm_eps=1e-6,
        tie_word_embeddings=True, attention_bias=False, use_sliding_window=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )


def make_tiny_qwen3(out_dir: str | Path, seed: int = 0, std: float = 0.08, **cfg_kw) -> Path:
    from transformers import Qwen3ForCausalLM
    out = Path(out_dir)
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(tiny_config(**cfg_kw)).float()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(0, std)
    model.save_pretrained(out, safe_serialization=True)
    return out


def load_hf(model_dir, dtype=torch.float32, attn: str = "eager"):
    from transformers import Qwen3ForCausalLM
    m = Qwen3ForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn)
    m.eval()
    return m


class LoraLinear(torch.nn.Module):
    """Reference LoRA wrapper for an HF nn.Linear, same formula as the engine."""

    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base, self.a, self.b, self.scale = base, a, b, scale

    def forward(self, x):
        return self.base(x) + (self.scale * torch.nn.functional.linear(
            torch.nn.functional.linear(x.float(), self.a), self.b)).to(x.dtype)


def attach_reference_lora(hf_model, lora) -> list[torch.Tensor]:
    """Wrap HF projections with copies of the engine's LoRA tensors. Returns the
    reference parameters in the same order as lora.parameters()."""
    params = []
    for i, layer in enumerate(hf_model.model.layers):
        for name, (a, b) in lora.layers[i].items():
            parent = layer.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
            ra = a.detach().clone().requires_grad_(True)
            rb = b.detach().clone().requires_grad_(True)
            setattr(parent, name, LoraLinear(getattr(parent, name), ra, rb, lora.scale))
            params += [ra, rb]
    for p in hf_model.parameters():
        p.requires_grad_(False)
    return params
