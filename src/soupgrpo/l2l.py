"""Layer-to-layer (L2L) streaming engine for Qwen3: one engine for prefill,
KV-cached decoding, forward-only log-probs and the LoRA training pass.

Core idea: weights live in CPU RAM; each decoder layer is copied to the GPU
once per pass and ALL data (every sequence / micro-batch) is pushed through it
before the next layer is needed. A ring of n GPU buffers plus a separate CUDA
copy stream prefetches layer k+1 while layer k computes.

Training pass (one call = forward + backward over a set of sequences):
  forward  (no grad): layer by layer; the INPUT of every layer for every
           micro-batch is saved ("boundary checkpoint") to CPU RAM.
  head     : final norm + LM head on scored positions only, in chunks of
           `logit_chunk` rows (full logits for 8k tokens would be ~5 GB);
           per-token loss -> backward gives dL/dh for the last hidden state.
  backward : layers in reverse; for each micro-batch, reload the boundary
           input, recompute the layer with autograd on, backprop the incoming
           gradient. LoRA A/B gradients accumulate; dL/dx flows down.
Frozen weights get no gradient; only LoRA parameters are trained.

Reference model for KL = same weights with the adapter switched off; it is
computed in the same forward, on the same loaded layer (no extra PCIe copies).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from .lora import LoRA
from .qwen3_math import KVSlot, build_attn_mask, decoder_layer, position_ids_from_mask, rms_norm, rope_cos_sin
from .weights import WeightStore


# ---------------------------------------------------------------- streaming

@dataclass
class StreamStats:
    copies: int = 0
    bytes: int = 0
    copy_wait_s: float = 0.0   # host-side time blocked waiting for a layer (CPU path only)
    per_layer_copies: list[int] = field(default_factory=list)


class LayerStreamer:
    def __init__(self, store: WeightStore, device: torch.device, n_buffers: int = 3, prefetch: bool = True,
                 resident_layers: int = 0, verify_copies: bool = False):
        assert n_buffers >= 2
        self.store, self.device = store, device
        self.cuda = device.type == "cuda"
        self.prefetch = prefetch and self.cuda
        self.verify = verify_copies
        n_layers = store.dims.n_layers
        # Resident layers: the LAST `resident_layers` layers stay on the GPU.
        self.resident: dict[int, dict[str, torch.Tensor]] = {}
        for li in range(n_layers - resident_layers, n_layers):
            self.resident[li] = store.views(store.layers[li].to(device))
        self.bufs = [torch.empty(store.layer_numel, dtype=store.dtype, device=device) for _ in range(n_buffers)]
        self.buf_views = [store.views(b) for b in self.bufs]
        self.slot_layer = [-1] * n_buffers
        self.ready_ev: list = [None] * n_buffers
        self.free_ev: list = [None] * n_buffers
        self.copy_stream = torch.cuda.Stream(device) if self.prefetch else None
        self.stats = StreamStats(per_layer_copies=[0] * n_layers)
        self._next_slot = 0

    def _issue(self, li: int) -> int | None:
        if li in self.resident:
            return None
        n = len(self.bufs)
        slot = self._next_slot
        self._next_slot = (slot + 1) % n
        if self.slot_layer[slot] == li:  # already there (e.g. tiny model, n >= layers)
            return slot
        src = self.store.layers[li]
        if self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                if self.free_ev[slot] is not None:
                    self.copy_stream.wait_event(self.free_ev[slot])
                self.bufs[slot].copy_(src, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.copy_stream)
                self.ready_ev[slot] = ev
        else:
            t0 = time.perf_counter()
            self.bufs[slot].copy_(src, non_blocking=self.cuda)
            self.stats.copy_wait_s += time.perf_counter() - t0
            self.ready_ev[slot] = None
        self.slot_layer[slot] = li
        self.stats.copies += 1
        self.stats.bytes += self.store.layer_bytes
        self.stats.per_layer_copies[li] += 1
        return slot

    def run(self, order):
        """Yield (layer_index, weight_views) in `order`, prefetching the next
        layer while the caller computes the current one. The caller must finish
        enqueuing all work for a layer before advancing the generator."""
        order = list(order)
        if not order:
            return
        slots = [None] * len(order)
        slots[0] = self._issue(order[0])
        for k, li in enumerate(order):
            if k + 1 < len(order):
                slots[k + 1] = self._issue(order[k + 1])
            slot = slots[k]
            if slot is None:
                yield li, self.resident[li]
                continue
            if self.ready_ev[slot] is not None:
                torch.cuda.current_stream(self.device).wait_event(self.ready_ev[slot])
            if self.verify:
                got = float(self.bufs[slot].float().sum())
                exp = self.store.checksums[li]
                if abs(got - exp) > 1e-3 * max(1.0, abs(exp)):
                    raise RuntimeError(f"layer buffer checksum mismatch: layer {li} slot {slot}: {got} vs {exp}")
            yield li, self.buf_views[slot]
            if self.cuda:
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(self.device))
                self.free_ev[slot] = ev


# ---------------------------------------------------------------- boundary store

class BoundaryStore:
    """CPU (pinned on CUDA hosts) storage for per-layer inputs, reused across
    passes so pinning cost is paid once.

    Host RAM matters here: for 48 sequences x 650 tokens x 2560 x fp32 x 36
    layers this is ~11.5 GB. The first Kaggle dry run (E14) ran the 31 GB host
    out of memory because growing the buffer briefly held three copies: the
    old pinned block (kept by PyTorch's pinned-memory cache), a pageable
    temporary, and its pinned copy. Now: free + release the pinned cache first,
    allocate pinned memory directly, and let the trainer reserve() the largest
    size once, so the buffer never grows during a run."""

    def __init__(self, pin: bool):
        self.pin = pin
        self.buf: torch.Tensor | None = None
        self.allocations = 0

    def _alloc(self, numel: int, dtype) -> None:
        self.buf = None
        release = getattr(torch._C, "_accelerator_emptyHostCache", None) or getattr(torch._C, "_host_emptyCache", None)
        if self.pin and release is not None:
            release()
        self.buf = torch.empty(numel, dtype=dtype, pin_memory=self.pin)
        self.allocations += 1

    def reserve(self, n_layers: int, n_elems: int, dtype) -> None:
        """Allocate for the largest pass up front (trainer start-up)."""
        if self.buf is None or self.buf.numel() < n_layers * n_elems or self.buf.dtype != dtype:
            self._alloc(n_layers * n_elems, dtype)

    def ensure(self, n_layers: int, n_elems: int, dtype) -> None:
        need = n_layers * n_elems
        if self.buf is None or self.buf.numel() < need or self.buf.dtype != dtype:
            self._alloc(need, dtype)
        self.n_elems = n_elems

    def nbytes(self) -> int:
        return 0 if self.buf is None else self.buf.numel() * self.buf.element_size()

    def slot(self, layer: int, offset: int, shape) -> torch.Tensor:
        n = 1
        for s in shape:
            n *= s
        base = layer * self.n_elems + offset
        return self.buf[base:base + n].view(shape)


# ---------------------------------------------------------------- engine

@dataclass
class GenOutput:
    tokens: list[int]
    finish: str               # "eos" or "length"
    logprobs: list[float]     # log p of each sampled token under the sampling distribution (T applied)


class L2LEngine:
    def __init__(self, store: WeightStore, device="cpu", lora: LoRA | None = None, n_buffers: int = 3,
                 prefetch: bool = True, resident_layers: int = 0, verify_copies: bool = False,
                 logit_chunk: int = 512, residual_fp32: bool | None = None):
        self.store = store
        self.d = store.dims
        self.device = torch.device(device)
        self.dtype = store.dtype
        # Residual stream dtype. fp16 (T4: no bf16) has a max of 65504, and the
        # residual stream of LLMs can grow past that, so in fp16 mode the hidden
        # state between sublayers is kept in fp32; matmuls still run in fp16.
        if residual_fp32 is None:
            residual_fp32 = store.dtype == torch.float16
        self.res_dtype = torch.float32 if residual_fp32 else store.dtype
        self.nonfinite_events = 0
        r = store.resident
        self.embed = r["model.embed_tokens.weight"].to(self.device)
        self.final_norm = r["model.norm.weight"].to(self.device)
        self.lm_head = r["lm_head.weight"].to(self.device) if "lm_head.weight" in r else self.embed
        self.streamer = LayerStreamer(store, self.device, n_buffers, prefetch, resident_layers, verify_copies)
        self.lora = lora
        self.logit_chunk = logit_chunk
        self.boundaries = BoundaryStore(pin=self.device.type == "cuda")
        self.timers: dict[str, float] = {}

    # ---------------- helpers
    def _lora(self, li: int, enabled: bool = True):
        if self.lora is None or not enabled:
            return None
        return self.lora.layer(li)

    @property
    def lora_scale(self) -> float:
        return self.lora.scale if self.lora is not None else 1.0

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        return F.linear(rms_norm(h, self.final_norm, self.d.rms_eps).to(self.lm_head.dtype), self.lm_head).float()

    def _embed(self, ids):
        return F.embedding(ids, self.embed).to(self.res_dtype)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _tick(self, name: str, t0: float) -> float:
        self._sync()
        t = time.perf_counter()
        self.timers[name] = self.timers.get(name, 0.0) + (t - t0)
        return t

    # ---------------- generation
    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new_tokens: int, temperature: float = 1.0,
                 stop_ids: tuple[int, ...] = (), pad_id: int = 0, batch_size: int = 64,
                 generator: torch.Generator | None = None, lora_enabled: bool = True) -> list[GenOutput]:
        outs: list[GenOutput] = []
        for s in range(0, len(prompts), batch_size):
            outs += self._generate_safe(prompts[s:s + batch_size], max_new_tokens, temperature,
                                        stop_ids, pad_id, generator, lora_enabled)
        return outs

    def _generate_safe(self, prompts, *args):
        """On CUDA out-of-memory, free the cache and run the batch as two halves.
        Counted in self.oom_splits (a run that needed splits reports it). The RNG
        stream then differs from an unsplit run, so splits are logged, not hidden."""
        try:
            return self._generate_batch(prompts, *args)
        except torch.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            self.oom_splits = getattr(self, "oom_splits", 0) + 1
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            h = len(prompts) // 2
            return self._generate_safe(prompts[:h], *args) + self._generate_safe(prompts[h:], *args)

    def _generate_batch(self, prompts, max_new, temperature, stop_ids, pad_id, gen, lora_enabled):
        d, dev = self.d, self.device
        B, P = len(prompts), max(len(p) for p in prompts)
        T_max = P + max_new
        ids = torch.full((B, P), pad_id, dtype=torch.long)
        attn = torch.zeros((B, T_max), dtype=torch.long)
        for b, p in enumerate(prompts):  # left padding: all prompts end at index P-1
            ids[b, P - len(p):] = torch.tensor(p)
            attn[b, P - len(p):P] = 1
        ids, attn = ids.to(dev), attn.to(dev)
        kv = torch.empty((d.n_layers, 2, B, d.n_kv_heads, T_max, d.head_dim), dtype=self.dtype, device=dev)
        stop = torch.tensor(list(stop_ids) or [-1], device=dev)

        t0 = time.perf_counter()
        # prefill
        x = self._embed(ids)
        pos = position_ids_from_mask(attn[:, :P])
        cos, sin = rope_cos_sin(pos, d.head_dim, d.rope_theta, self.dtype)
        mask = build_attn_mask(attn[:, :P], P, P)
        for li, w in self.streamer.run(range(d.n_layers)):
            x = decoder_layer(x, w, d, cos, sin, mask, self._lora(li, lora_enabled), self.lora_scale,
                              KVSlot(kv[li, 0], kv[li, 1]), 0)
        logits = self.head_logits(x[:, -1])
        t0 = self._tick("prefill", t0)

        tokens = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
        logps = torch.zeros((B, max_new), dtype=torch.float32, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        lengths = torch.full((B,), max_new, dtype=torch.long, device=dev)
        for t in range(max_new):
            bad = ~torch.isfinite(logits).all(-1) & ~finished
            if bool(bad.any()):  # fp16 overflow guard: count it, never sample from NaN/inf
                self.nonfinite_events += int(bad.sum())
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            if temperature == 0:
                tok = logits.argmax(-1)
                lp = torch.log_softmax(logits, -1).gather(-1, tok[:, None])[:, 0]
            else:
                lsm = torch.log_softmax(logits / temperature, -1)
                tok = torch.multinomial(lsm.exp(), 1, generator=gen)[:, 0]
                lp = lsm.gather(-1, tok[:, None])[:, 0]
            tok = torch.where(finished, torch.full_like(tok, pad_id), tok)
            tokens[:, t] = tok
            logps[:, t] = torch.where(finished, torch.zeros_like(lp), lp)
            just = ~finished & torch.isin(tok, stop)
            lengths = torch.where(just, torch.full_like(lengths, t + 1), lengths)
            finished |= just
            if bool(finished.all()) or t == max_new - 1:
                break
            # feed tok at KV position P + t
            kp = P + t
            attn[:, kp] = 1  # finished rows keep decoding pad tokens; their outputs are discarded
            x = self._embed(tok[:, None])
            pos = position_ids_from_mask(attn[:, :kp + 1])[:, -1:]
            cos, sin = rope_cos_sin(pos, d.head_dim, d.rope_theta, self.dtype)
            mask = build_attn_mask(attn[:, :kp + 1], 1, kp + 1)
            for li, w in self.streamer.run(range(d.n_layers)):
                x = decoder_layer(x, w, d, cos, sin, mask, self._lora(li, lora_enabled), self.lora_scale,
                                  KVSlot(kv[li, 0], kv[li, 1]), kp)
            logits = self.head_logits(x[:, -1])
        self._tick("decode", t0)

        tokens, logps, lengths, finished = tokens.cpu(), logps.cpu(), lengths.cpu(), finished.cpu()
        out = []
        for b in range(B):
            n = int(lengths[b])
            out.append(GenOutput(tokens[b, :n].tolist(), "eos" if bool(finished[b]) else "length",
                                 logps[b, :n].tolist()))
        return out

    # ---------------- forward over full sequences
    def _prep(self, ids, attn, micro_batch):
        d = self.d
        mbs = []
        for s in range(0, ids.shape[0], micro_batch):
            i = ids[s:s + micro_batch].to(self.device)
            a = attn[s:s + micro_batch].to(self.device)
            pos = position_ids_from_mask(a)
            cos, sin = rope_cos_sin(pos, d.head_dim, d.rope_theta, self.dtype)
            mbs.append(dict(ids=i, attn=a, cos=cos, sin=sin, mask=build_attn_mask(a, a.shape[1], a.shape[1])))
        return mbs

    @torch.no_grad()
    def _forward(self, mbs, save_boundaries: bool, with_ref: bool, lora_enabled: bool = True):
        d = self.d
        xs = [self._embed(m["ids"]) for m in mbs]
        xr = [x.clone() for x in xs] if with_ref else None
        offsets = []
        if save_boundaries:
            off = 0
            for x in xs:
                offsets.append(off)
                off += x.numel()
            self.boundaries.ensure(d.n_layers, off, self.res_dtype)
        for li, w in self.streamer.run(range(d.n_layers)):
            for j, m in enumerate(mbs):
                if save_boundaries:
                    self.boundaries.slot(li, offsets[j], xs[j].shape).copy_(xs[j], non_blocking=True)
                xs[j] = decoder_layer(xs[j], w, d, m["cos"], m["sin"], m["mask"],
                                      self._lora(li, lora_enabled), self.lora_scale)
                if with_ref:
                    xr[j] = decoder_layer(xr[j], w, d, m["cos"], m["sin"], m["mask"], None)
        return xs, xr, offsets

    @staticmethod
    def _scored_index(attn, loss_mask):
        """Positions t>=1 with loss_mask[:, t] == 1 AND a real token at t-1 are
        scored: logp(ids[t] | h[t-1]). Requiring attn[t-1] stops a token right
        after padding from being scored off a pad position's hidden state."""
        lm = loss_mask.bool() & attn.bool()
        lm[:, 1:] &= attn[:, :-1].bool()
        lm[:, 0] = False
        b, t = lm.nonzero(as_tuple=True)
        return b, t

    @torch.no_grad()
    def logprobs(self, ids: torch.Tensor, attn: torch.Tensor, loss_mask: torch.Tensor, micro_batch: int = 8,
                 lora_enabled: bool = True) -> torch.Tensor:
        """Per-token log-probs [B, T] (0 where not scored). Forward only."""
        mbs = self._prep(ids, attn, micro_batch)
        xs, _, _ = self._forward(mbs, save_boundaries=False, with_ref=False, lora_enabled=lora_enabled)
        out = torch.zeros(ids.shape, dtype=torch.float32)
        for j, m in enumerate(mbs):
            s = j * micro_batch
            b, t = self._scored_index(m["attn"], loss_mask[s:s + micro_batch].to(self.device))
            for c in range(0, len(b), self.logit_chunk):
                bb, tt = b[c:c + self.logit_chunk], t[c:c + self.logit_chunk]
                lp = torch.log_softmax(self.head_logits(xs[j][bb, tt - 1]), -1)
                out[s + bb.cpu(), tt.cpu()] = lp.gather(-1, m["ids"][bb, tt][:, None])[:, 0].cpu()
        return out

    @torch.no_grad()
    def full_logits(self, ids, attn, micro_batch: int = 8, lora_enabled: bool = True) -> torch.Tensor:
        """All logits [B, T, V] in fp32. Small inputs only (tests, merge check)."""
        mbs = self._prep(ids, attn, micro_batch)
        xs, _, _ = self._forward(mbs, save_boundaries=False, with_ref=False, lora_enabled=lora_enabled)
        return torch.cat([self.head_logits(x).cpu() for x in xs])

    # ---------------- training pass
    def train_pass(self, ids: torch.Tensor, attn: torch.Tensor, loss_mask: torch.Tensor,
                   loss_fn: Callable[[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor], torch.Tensor],
                   micro_batch: int = 8, with_ref: bool = False) -> dict:
        """Forward + backward over the given sequences. LoRA .grad accumulates
        (call zero_grad / optimizer.step outside).

        loss_fn(logp, ref_logp_or_None, b_idx, t_idx) -> scalar for one chunk of
        scored tokens (b_idx/t_idx index into this call's [B, T]). The total loss
        is the sum over chunks, so loss_fn must be a sum of per-token terms.
        """
        assert self.lora is not None and self.lora.enabled
        d = self.d
        t0 = time.perf_counter()
        mbs = self._prep(ids, attn, micro_batch)
        xs, xr, offsets = self._forward(mbs, save_boundaries=True, with_ref=with_ref)
        t0 = self._tick("train_forward", t0)

        # head: chunked log-probs and dL/dh of the last hidden state
        grads = [torch.zeros_like(x) for x in xs]
        total_loss, n_tok = 0.0, 0
        logp_all = torch.zeros(ids.shape, dtype=torch.float32)
        ref_all = torch.zeros(ids.shape, dtype=torch.float32) if with_ref else None
        for j, m in enumerate(mbs):
            s = j * micro_batch
            b, t = self._scored_index(m["attn"], loss_mask[s:s + micro_batch].to(self.device))
            for c in range(0, len(b), self.logit_chunk):
                bb, tt = b[c:c + self.logit_chunk], t[c:c + self.logit_chunk]
                tgt = m["ids"][bb, tt][:, None]
                rows = xs[j][bb, tt - 1].detach().requires_grad_(True)
                with torch.enable_grad():
                    lp = torch.log_softmax(self.head_logits(rows), -1).gather(-1, tgt)[:, 0]
                    ref_lp = None
                    if with_ref:
                        with torch.no_grad():
                            ref_lp = torch.log_softmax(self.head_logits(xr[j][bb, tt - 1]), -1).gather(-1, tgt)[:, 0]
                    if not bool(torch.isfinite(lp).all()):
                        self.nonfinite_events += int((~torch.isfinite(lp)).sum())
                        raise FloatingPointError("non-finite log-probs in training pass (fp16 overflow?)")
                    loss = loss_fn(lp, ref_lp, s + bb, tt)
                    loss.backward()
                grads[j].index_put_((bb, tt - 1), rows.grad.to(grads[j].dtype), accumulate=True)
                total_loss += float(loss.detach())
                n_tok += len(bb)
                logp_all[s + bb.cpu(), tt.cpu()] = lp.detach().cpu()
                if with_ref:
                    ref_all[s + bb.cpu(), tt.cpu()] = ref_lp.cpu()
        del xs, xr
        t0 = self._tick("train_head", t0)

        # backward: reverse layers, recompute each from its saved input
        for li, w in self.streamer.run(reversed(range(d.n_layers))):
            lora = self.lora.layer(li)
            for j, m in enumerate(mbs):
                x = self.boundaries.slot(li, offsets[j], grads[j].shape).to(self.device, non_blocking=True)
                x.requires_grad_(True)
                with torch.enable_grad():
                    out = decoder_layer(x, w, d, m["cos"], m["sin"], m["mask"], lora, self.lora_scale)
                    out.backward(grads[j])
                grads[j] = x.grad
        self._tick("train_backward", t0)
        return {"loss": total_loss, "n_tokens": n_tok, "logp": logp_all, "ref_logp": ref_all}
