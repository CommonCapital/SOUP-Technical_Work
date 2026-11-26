"""DFlash speculative decoding (arXiv 2602.06036) on top of the L2L engine.

Why it fits L2L: one decoding pass costs one full stream of the 36 layers over
PCIe (~0.6 s on the T4) no matter how many tokens per sequence it processes.
A drafter that proposes 15 tokens which the target checks in ONE pass turns
"1 token per pass" into "accepted + 1 tokens per pass".

Drafter (checkpoint z-lab/Qwen3-4B-DFlash-b16; re-implemented here from the
published architecture, none of its code is run):
  * 5 Qwen3-style layers, resident on the GPU (~1.1 GB), block size 16,
    mask token 151669; shares the target's embedding and LM head.
  * Context = outputs of target layers [1, 9, 17, 25, 33], concatenated,
    -> fc (12800 -> 2560, no bias) -> RMSNorm ("hidden_norm"). Every drafter
    layer turns that context into its own K/V (k_proj, k_norm, RoPE / v_proj)
    and caches it ("KV injection"); the context never enters the residual stream.
  * Block = [last committed token, mask x 15], embedded with the target's
    embedding; queries of the block attend to all cached context plus the whole
    block (NOT causal); output norm -> target LM head -> 15 draft tokens.
  * fc is applied as a sum of five 2560x2560 blocks while the target streams
    past each feature layer, so the five hidden states are never stored; fc
    runs in fp32 because raw residual-stream values can exceed fp16's range.

Verification on the target (one L2L pass over 16 tokens per row):
  * greedy: accept the longest prefix where draft == target argmax, then add the
    target's own token at the first mismatch ("bonus"). Output is token-identical
    to plain greedy decoding up to floating-point near-ties.
  * sampling (T > 0): standard speculative sampling (accept d with prob
    min(1, p(d)/q(d)), on rejection sample from normalise(max(p - q, 0))), which
    leaves the target's distribution unchanged. This is valid although the
    drafter proposes all 15 tokens at once: q_i does not depend on d_<i, so it
    IS the proposal distribution of d_i given the prefix.
  * rows accept different numbers of tokens, so KV writes are per row
    (qwen3_math.decoder_layer with kv_pos = (start[B], kv_len)); rejected slots
    are simply overwritten by the next pass (causal mask hides them before that).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .l2l import GenOutput, L2LEngine
from .qwen3_math import KVSlot, Qwen3Dims, apply_rope, build_attn_mask, decoder_layer, position_ids_from_mask, \
    rms_norm, rope_cos_sin

DRAFT_TENSORS = {
    "input_layernorm.weight": "ln1", "post_attention_layernorm.weight": "ln2",
    "self_attn.q_proj.weight": "q_proj", "self_attn.k_proj.weight": "k_proj", "self_attn.v_proj.weight": "v_proj",
    "self_attn.o_proj.weight": "o_proj", "self_attn.q_norm.weight": "q_norm", "self_attn.k_norm.weight": "k_norm",
    "mlp.gate_proj.weight": "gate_proj", "mlp.up_proj.weight": "up_proj", "mlp.down_proj.weight": "down_proj",
}


# ---------------------------------------------------------------- drafter

class Drafter:
    def __init__(self, dims: Qwen3Dims, layers: list[dict], fc: torch.Tensor, hidden_norm: torch.Tensor,
                 norm: torch.Tensor, target_layer_ids: list[int], block_size: int, mask_token_id: int,
                 device, dtype):
        self.d, self.device, self.dtype = dims, torch.device(device), dtype
        self.layers = [{k: v.to(self.device, dtype) for k, v in w.items()} for w in layers]
        H = dims.hidden
        fc = fc.to(self.device, torch.float32)
        assert fc.shape == (H, H * len(target_layer_ids)), fc.shape
        # fc(concat(h_1..h_5)) = sum_k h_k @ fc[:, kH:(k+1)H]^T
        self.fc_blocks = {li: fc[:, k * H:(k + 1) * H].contiguous() for k, li in enumerate(target_layer_ids)}
        self.hidden_norm = hidden_norm.to(self.device, torch.float32)
        self.norm = norm.to(self.device, dtype)
        self.target_layer_ids = list(target_layer_ids)
        self.block_size, self.mask_token_id = block_size, mask_token_id
        self.max_abs_weight = max(float(v.abs().max()) for w in layers for v in w.values())

    # -------- loading
    @classmethod
    def load(cls, path: str | Path, device, dtype=torch.float16) -> "Drafter":
        from safetensors import safe_open
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        dims = Qwen3Dims.from_hf_config(cfg)
        dcfg = cfg.get("dflash_config", {})
        tl = dcfg.get("target_layer_ids") or build_target_layer_ids(cfg["num_target_layers"], dims.n_layers)
        layers = [dict() for _ in range(dims.n_layers)]
        top = {}
        files = sorted(path.glob("*.safetensors"))
        for f in files:
            with safe_open(str(f), framework="pt") as sf:
                for key in sf.keys():
                    k = key[len("model."):] if key.startswith("model.") else key
                    t = sf.get_tensor(key)
                    if k.startswith("layers."):
                        li, suffix = k[len("layers."):].split(".", 1)
                        if suffix not in DRAFT_TENSORS:
                            raise KeyError(f"unexpected drafter tensor {key}")
                        layers[int(li)][DRAFT_TENSORS[suffix]] = t
                    elif k in ("fc.weight", "hidden_norm.weight", "norm.weight"):
                        top[k] = t
                    else:
                        raise KeyError(f"unexpected drafter tensor {key} (shape {tuple(t.shape)})")
        for i, w in enumerate(layers):
            missing = set(DRAFT_TENSORS.values()) - set(w)
            if missing:
                raise KeyError(f"drafter layer {i} missing {sorted(missing)}")
        return cls(dims, layers, top["fc.weight"], top["hidden_norm.weight"], top["norm.weight"], tl,
                   cfg.get("block_size", 16), dcfg.get("mask_token_id", 151669), device, dtype)

    @classmethod
    def random(cls, target: Qwen3Dims, n_layers: int, target_layer_ids, block_size: int, mask_token_id: int,
               device="cpu", dtype=torch.float32, seed: int = 0, std: float = 0.05) -> "Drafter":
        """Random drafter with the target's dims (tests only)."""
        g = torch.Generator().manual_seed(seed)
        d = target
        io = d.proj_io()
        def rnd(*s):
            return torch.randn(*s, generator=g) * std
        layers = []
        for _ in range(n_layers):
            w = {n: rnd(o, i) for n, (i, o) in io.items()}
            w.update(ln1=torch.ones(d.hidden), ln2=torch.ones(d.hidden), q_norm=torch.ones(d.head_dim),
                     k_norm=torch.ones(d.head_dim))
            layers.append(w)
        fc = rnd(d.hidden, d.hidden * len(target_layer_ids))
        return cls(d, layers, fc, torch.ones(d.hidden), torch.ones(d.hidden), list(target_layer_ids),
                   block_size, mask_token_id, device, dtype)

    def nbytes(self) -> int:
        n = sum(v.numel() * v.element_size() for w in self.layers for v in w.values())
        return n + sum(v.numel() * 4 for v in self.fc_blocks.values()) + self.norm.numel() * self.norm.element_size()

    # -------- context (target features -> per-layer K/V cache)
    def new_cache(self, B: int, T_max: int):
        d = self.d
        return torch.zeros((len(self.layers), 2, B, d.n_kv_heads, T_max, d.head_dim), dtype=self.dtype,
                           device=self.device)

    def add_context(self, cache, ctx_acc: torch.Tensor, start: torch.Tensor, pos: torch.Tensor):
        """ctx_acc [B, n, H] fp32 = sum of fc blocks applied to the feature layers.
        Writes the K/V of these n context positions into cache slots start[b] ... start[b]+n-1."""
        d = self.d
        B, n, _ = ctx_acc.shape
        ctx = rms_norm(ctx_acc, self.hidden_norm, d.rms_eps).to(self.dtype)
        cos, sin = rope_cos_sin(pos, d.head_dim, d.rope_theta, self.dtype)
        idx = start[:, None] + torch.arange(n, device=start.device)[None]
        rows = torch.arange(B, device=start.device)[:, None]
        for li, w in enumerate(self.layers):
            k = F.linear(ctx, w["k_proj"]).view(B, n, d.n_kv_heads, d.head_dim)
            k = apply_rope(rms_norm(k, w["k_norm"], d.rms_eps).transpose(1, 2), cos, sin)
            v = F.linear(ctx, w["v_proj"]).view(B, n, d.n_kv_heads, d.head_dim)
            cache[li, 0][rows, :, idx] = k.transpose(1, 2)
            cache[li, 1][rows, :, idx] = v

    # -------- one draft block
    def forward_block(self, cache, x: torch.Tensor, pos: torch.Tensor, ctx_valid: torch.Tensor) -> torch.Tensor:
        """x [B, S, H] embedded block, pos [B, S], ctx_valid bool [B, L] (cache slots to attend).
        Returns the normed drafter hidden states [B, S, H] (feed to the target LM head)."""
        d = self.d
        B, S, _ = x.shape
        L = ctx_valid.shape[1]
        cos, sin = rope_cos_sin(pos, d.head_dim, d.rope_theta, self.dtype)
        mask = torch.cat([ctx_valid[:, None, None, :].expand(B, 1, S, L),
                          torch.ones((B, 1, S, S), dtype=torch.bool, device=x.device)], dim=-1)
        x = x.float()
        for li, w in enumerate(self.layers):
            h = rms_norm(x, w["ln1"], d.rms_eps).to(self.dtype)
            q = F.linear(h, w["q_proj"]).view(B, S, d.n_heads, d.head_dim)
            k = F.linear(h, w["k_proj"]).view(B, S, d.n_kv_heads, d.head_dim)
            v = F.linear(h, w["v_proj"]).view(B, S, d.n_kv_heads, d.head_dim).transpose(1, 2)
            q = apply_rope(rms_norm(q, w["q_norm"], d.rms_eps).transpose(1, 2), cos, sin)
            k = apply_rope(rms_norm(k, w["k_norm"], d.rms_eps).transpose(1, 2), cos, sin)
            K = torch.cat([cache[li, 0][:, :, :L], k], dim=2)
            V = torch.cat([cache[li, 1][:, :, :L], v], dim=2)
            o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, scale=d.head_dim ** -0.5,
                                               enable_gqa=d.n_heads != d.n_kv_heads)
            x = x + F.linear(o.transpose(1, 2).reshape(B, S, -1), w["o_proj"]).float()
            h = rms_norm(x, w["ln2"], d.rms_eps).to(self.dtype)
            x = x + F.linear(F.silu(F.linear(h, w["gate_proj"])) * F.linear(h, w["up_proj"]), w["down_proj"]).float()
        return rms_norm(x, self.norm.float(), d.rms_eps)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start, end = 1, num_target_layers - 3
    return [int(round(start + i * (end - start) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


# ---------------------------------------------------------------- acceptance rules

def speculative_accept(p: torch.Tensor, q: torch.Tensor, draft: torch.Tensor, u: torch.Tensor):
    """Standard speculative sampling for a block, one row per batch entry.
    p [B, k+1, V]: target probs for draft positions 1..k and the position after;
    q [B, k, V]: drafter probs; draft [B, k]; u [B, k] uniforms.
    Returns (n_accepted [B], residual distribution for the next token [B, V])."""
    B, k = draft.shape
    pd = p[:, :k].gather(-1, draft[..., None])[..., 0]
    qd = q.gather(-1, draft[..., None])[..., 0]
    ok = (u * qd < pd).long().cumprod(-1)
    n = ok.sum(-1)
    rows = torch.arange(B, device=p.device)
    full = n == k
    nc = n.clamp(max=k - 1)
    resid = (p[rows, nc] - q[rows, nc]).clamp_min(0)
    tot = resid.sum(-1, keepdim=True)
    resid = torch.where(tot > 0, resid / tot.clamp_min(1e-30), p[rows, nc])
    nxt = torch.where(full[:, None], p[rows, n.clamp(max=k)], resid)
    return n, nxt


# ---------------------------------------------------------------- generation

@dataclass
class DFlashStats:
    passes: int = 0                 # target verification passes (each = one full layer stream)
    accept_lens: list[int] = field(default_factory=list)   # tokens committed per (row, pass), incl. the bonus
    timers: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        a = self.accept_lens
        return {"passes": self.passes, "mean_accept_len": (sum(a) / len(a)) if a else None,
                "accept_len_hist": {i: a.count(i) for i in sorted(set(a))}, "timers": dict(self.timers)}


class DFlashGenerator:
    """Speculative generation for an L2LEngine (any LoRA state) with a resident drafter."""

    def __init__(self, eng: L2LEngine, drafter: Drafter, mode: str = "exact", logit_rows: int = 8):
        assert mode in ("exact", "accept_all")      # accept_all = deliberately biased (test power check only)
        self.eng, self.dr, self.mode, self.logit_rows = eng, drafter, mode, logit_rows
        self.feat_set = set(drafter.target_layer_ids)
        self.stats = DFlashStats()

    def _tick(self, name, t0):
        self.eng._sync()
        t = time.perf_counter()
        self.stats.timers[name] = self.stats.timers.get(name, 0.0) + t - t0
        return t

    @torch.no_grad()
    def generate(self, prompts, max_new, temperature=0.0, stop_ids=(), pad_id=0, batch_size=32, generator=None,
                 lora_enabled=True, draft_fn=None) -> list[GenOutput]:
        outs = []
        for s in range(0, len(prompts), batch_size):
            outs += self._safe(prompts[s:s + batch_size], max_new, temperature, stop_ids, pad_id, generator,
                               lora_enabled, draft_fn)
        return outs

    def _safe(self, prompts, *args):
        """Same OOM policy as the plain engine: split the batch in two, count it."""
        try:
            return self._batch(prompts, *args)
        except torch.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            self.oom_splits = getattr(self, "oom_splits", 0) + 1
            torch.cuda.empty_cache()
            h = len(prompts) // 2
            return self._safe(prompts[:h], *args) + self._safe(prompts[h:], *args)

    def _target_pass(self, x, cos, sin, mask, kv, kv_pos, lora_enabled, feat_acc):
        e = self.eng
        for li, w in e.streamer.run(range(e.d.n_layers)):
            x = decoder_layer(x, w, e.d, cos, sin, mask, e._lora(li, lora_enabled), e.lora_scale,
                              KVSlot(kv[li, 0], kv[li, 1]), kv_pos)
            if li in self.feat_set:
                feat_acc += F.linear(x.float(), self.dr.fc_blocks[li])
        return x

    def _batch(self, prompts, max_new, T, stop_ids, pad_id, gen, lora_enabled, draft_fn):
        e, dr, d = self.eng, self.dr, self.eng.d
        dev, S = e.device, dr.block_size
        B, P = len(prompts), max(len(p) for p in prompts)
        T_max = P + max_new + S + 1
        ids = torch.full((B, P), pad_id, dtype=torch.long)
        attn = torch.zeros((B, P), dtype=torch.long)
        for b, p in enumerate(prompts):
            ids[b, P - len(p):] = torch.tensor(p)
            attn[b, P - len(p):] = 1
        ids, attn = ids.to(dev), attn.to(dev)
        pad = (P - attn.sum(-1)).long()                              # left padding per row
        # zeros, not empty: the verify pass reads slots up to the LONGEST row, so shorter rows see slots
        # nobody wrote. They are masked, but SDPA adds the mask (-inf) to q.k, and NaN garbage + -inf = NaN
        # (found in a local test: intermittent NaN drafts, depending on what the allocator handed out).
        kv = torch.zeros((d.n_layers, 2, B, d.n_kv_heads, T_max, d.head_dim), dtype=e.dtype, device=dev)
        dcache = dr.new_cache(B, T_max)
        stop = torch.tensor(list(stop_ids) or [-1], device=dev)
        rows = torch.arange(B, device=dev)
        ar = torch.arange(S, device=dev)
        slots = torch.arange(T_max, device=dev)

        # ---- prefill (also gives the drafter its first context: every prompt position)
        t0 = time.perf_counter()
        pos = position_ids_from_mask(attn)
        cos, sin = rope_cos_sin(pos, d.head_dim, d.rope_theta, e.dtype)
        feat = torch.zeros((B, P, d.hidden), dtype=torch.float32, device=dev)
        x = self._target_pass(e._embed(ids), cos, sin, build_attn_mask(attn, P, P), kv, 0, lora_enabled, feat)
        logits = e.head_logits(x[:, -1])
        dr.add_context(dcache, feat, torch.zeros(B, dtype=torch.long, device=dev), pos)
        del feat
        self.stats.passes += 1
        t0 = self._tick("prefill", t0)

        out = torch.full((B, max_new + S + 1), pad_id, dtype=torch.long, device=dev)
        lps = torch.zeros((B, max_new + S + 1), dtype=torch.float32, device=dev)
        tok, lp = self._pick(logits, T, gen)
        out[:, 0], lps[:, 0] = tok, lp
        n = torch.ones(B, dtype=torch.long, device=dev)               # committed tokens per row
        done = torch.isin(tok, stop) | (n >= max_new)
        ctx_len = torch.full((B,), P, dtype=torch.long, device=dev)   # drafter context = slots [0, ctx_len)

        while not bool(done.all()):
            start = P + n - 1                                          # slot of the last committed token
            blk = torch.full((B, S), dr.mask_token_id, dtype=torch.long, device=dev)
            blk[:, 0] = out[rows, n - 1]
            bpos = start[:, None] + ar[None] - pad[:, None]
            # ---- draft
            L = int(ctx_len.max())
            ctx_valid = (slots[None, :L] < ctx_len[:, None]) & (slots[None, :L] >= pad[:, None])
            hD = dr.forward_block(dcache, e._embed(blk).to(dr.dtype), bpos, ctx_valid)[:, 1:]
            draft, q = self._draft_tokens(hD, T, gen) if draft_fn is None else draft_fn(out, n, S - 1)
            blk[:, 1:] = draft
            t0 = self._tick("draft", t0)
            # ---- verify: one target pass over 16 tokens per row
            kv_len = int(start.max()) + S
            qslot = start[:, None] + ar[None]                          # [B, S]
            mask = (slots[None, None, :kv_len] <= qslot[:, :, None]) & (slots[None, None, :kv_len] >= pad[:, None, None])
            cos, sin = rope_cos_sin(bpos, d.head_dim, d.rope_theta, e.dtype)
            feat = torch.zeros((B, S, d.hidden), dtype=torch.float32, device=dev)
            x = self._target_pass(e._embed(blk), cos, sin, mask[:, None], kv, (start, kv_len), lora_enabled, feat)
            self.stats.passes += 1
            t0 = self._tick("verify", t0)
            # ---- accept
            acc, nxt_tok, new_lp = self._accept(x, draft, q, hD, T, gen)
            # new tokens = draft[:acc] + nxt_tok; their log-probs = new_lp[:, :acc+1]
            n_new = torch.where(done, torch.zeros_like(acc), acc + 1)
            cand = torch.cat([draft, torch.zeros((B, 1), dtype=torch.long, device=dev)], 1)
            cand[rows, acc] = nxt_tok
            # stop at the first stop token, and at max_new
            is_stop = torch.isin(cand, stop) & (ar[None, :] < (acc + 1)[:, None])
            first_stop = torch.where(is_stop.any(-1), is_stop.float().argmax(-1) + 1, acc + 1)
            n_new = torch.minimum(torch.minimum(n_new, first_stop), (max_new - n).clamp(min=0))
            n_new = torch.where(done, torch.zeros_like(n_new), n_new)
            wmask = ar[None, :] < n_new[:, None]
            widx = (n[:, None] + ar[None, :]).clamp(max=out.shape[1] - 1)
            out[rows[:, None], widx] = torch.where(wmask, cand, out[rows[:, None], widx])
            lps[rows[:, None], widx] = torch.where(wmask, new_lp, lps[rows[:, None], widx])
            self.stats.accept_lens += n_new[~done].tolist()
            hit_stop = is_stop.any(-1) & (first_stop <= n_new) & ~done
            # drafter context: features of the committed block positions 0..acc (slots start..start+acc)
            dr.add_context(dcache, feat, start, bpos)
            ctx_len = torch.where(done, ctx_len, start + n_new)
            n = n + n_new
            done = done | hit_stop | (n >= max_new)
            t0 = self._tick("accept", t0)

        res = []
        out, lps, n = out.cpu(), lps.cpu(), n.cpu()
        for b in range(B):
            k = int(n[b])
            toks = out[b, :k].tolist()
            fin = "eos" if any(t in stop_ids for t in toks) else "length"
            res.append(GenOutput(toks, fin, lps[b, :k].tolist()))
        return res

    # -------- token choice helpers (chunked over rows: full-vocab logits are 0.6 MB per row-position)
    @staticmethod
    def _pick(logits, T, gen):
        if T == 0:
            tok = logits.argmax(-1)
            return tok, torch.log_softmax(logits, -1).gather(-1, tok[:, None])[:, 0]
        lsm = torch.log_softmax(logits / T, -1)
        tok = torch.multinomial(lsm.exp(), 1, generator=gen)[:, 0]
        return tok, lsm.gather(-1, tok[:, None])[:, 0]

    def _draft_logits(self, h):
        e = self.eng
        return F.linear(h.to(e.lm_head.dtype), e.lm_head).float()

    def _draft_tokens(self, hD, T, gen):
        B, k, _ = hD.shape
        toks = torch.empty((B, k), dtype=torch.long, device=hD.device)
        for s in range(0, B, self.logit_rows):
            lg = self._draft_logits(hD[s:s + self.logit_rows])
            if T == 0:
                toks[s:s + self.logit_rows] = lg.argmax(-1)
            else:
                pr = torch.softmax(lg / T, -1)
                toks[s:s + self.logit_rows] = torch.multinomial(pr.flatten(0, 1), 1, generator=gen).view(-1, k)
        return toks, None

    def _accept(self, x, draft, _q, hD, T, gen):
        """x [B, S, H] target hidden states of the block. Returns (n_accepted [B],
        next token [B], log-probs [B, S] of the would-be committed tokens draft[:n] + next)."""
        e = self.eng
        B, k = draft.shape
        acc = torch.empty(B, dtype=torch.long, device=x.device)
        nxt = torch.empty(B, dtype=torch.long, device=x.device)
        lp = torch.zeros((B, k + 1), dtype=torch.float32, device=x.device)
        for s in range(0, B, self.logit_rows):
            sl = slice(s, s + self.logit_rows)
            lg = e.head_logits(x[sl])                                 # [b, S, V]
            bad = ~torch.isfinite(lg).all(-1)
            if bool(bad.any()):
                e.nonfinite_events += int(bad.sum())
                lg = torch.nan_to_num(lg, nan=0.0, posinf=1e4, neginf=-1e4)
            dr_ = draft[sl]
            b = dr_.shape[0]
            rr = torch.arange(b, device=x.device)
            if T == 0:
                post = lg.argmax(-1)                                   # [b, S]
                a = (dr_ == post[:, :k]).long().cumprod(-1).sum(-1)
                if self.mode == "accept_all":
                    a = torch.full_like(a, k)
                nt = post[rr, a]
                lsm = torch.log_softmax(lg, -1)
            else:
                lsm = torch.log_softmax(lg / T, -1)
                p = lsm.exp()
                if self.mode == "accept_all":
                    a = torch.full((b,), k, dtype=torch.long, device=x.device)
                    nt = torch.multinomial(p[:, k], 1, generator=gen)[:, 0]
                else:
                    q = torch.softmax(self._draft_logits(hD[sl]) / T, -1)
                    u = torch.rand((b, k), generator=gen, device=x.device)
                    a, dist = speculative_accept(p, q, dr_, u)
                    nt = torch.multinomial(dist, 1, generator=gen)[:, 0]
            cand = torch.cat([dr_, torch.zeros((b, 1), dtype=torch.long, device=x.device)], 1)
            cand[rr, a] = nt
            lp[sl] = lsm.gather(-1, cand[..., None])[..., 0]
            acc[sl], nxt[sl] = a, nt
        return acc, nxt, lp


def write_random_drafter(out_dir: str | Path, target_dir: str | Path, n_layers: int = 2,
                         target_layer_ids=(1, 2), block_size: int = 16, mask_token_id: int = 7, seed: int = 0) -> Path:
    """Write a random drafter in the checkpoint's file format (config.json with
    dflash_config + model.safetensors with 'layers.N.self_attn.q_proj.weight'
    names), so local tests exercise the same loader as the real checkpoint."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tcfg = json.loads((Path(target_dir) / "config.json").read_text())
    dims = Qwen3Dims.from_hf_config(tcfg)
    dr = Drafter.random(dims, n_layers, list(target_layer_ids), block_size, mask_token_id, seed=seed)
    inv = {v: k for k, v in DRAFT_TENSORS.items()}
    t = {f"layers.{i}.{inv[k]}": v.contiguous() for i, w in enumerate(dr.layers) for k, v in w.items()}
    t["fc.weight"] = torch.cat([dr.fc_blocks[li] for li in dr.target_layer_ids], 1).contiguous()
    t["hidden_norm.weight"], t["norm.weight"] = dr.hidden_norm.contiguous(), dr.norm.contiguous()
    save_file(t, str(out / "model.safetensors"))
    cfg = {k: tcfg[k] for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
                                "head_dim", "vocab_size", "rms_norm_eps", "tie_word_embeddings") if k in tcfg}
    cfg.update(num_hidden_layers=n_layers, rope_theta=dims.rope_theta, block_size=block_size,
               num_target_layers=dims.n_layers,
               dflash_config={"mask_token_id": mask_token_id, "target_layer_ids": list(target_layer_ids)})
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    return out
