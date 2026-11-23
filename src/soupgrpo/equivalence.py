"""L2L equivalence checks (H17, H18, H4) against plain HF transformers Qwen3.

Metric: normalised max error  e(X, Ref) = max|X - Ref| / max|Ref|, computed on
real (non-pad) positions. Plain elementwise relative error is undefined near 0,
so the max is normalised by the reference's scale. Relative Frobenius error is
reported too.

Decision rules (pre-registered, predictions/PREDICTIONS.md):
  fp32: logits and every LoRA gradient e <= 1e-5
  bf16: e(L2L, plain bf16) <= 1.5 x e(plain bf16 eager, plain bf16 sdpa)  (Soup A3-07)
        reported for both metrics; the noise floor is measured, not assumed
  KV decoding: greedy tokens identical on 50 prompts (fp32)
  copies: = n_layers x passes
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .l2l import L2LEngine
from .lora import LoRA
from .qwen3_math import position_ids_from_mask
from .tiny import attach_reference_lora, load_hf, make_tiny_qwen3
from .weights import WeightStore


def nmax(x, ref) -> float:
    return float((x - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def rfro(x, ref) -> float:
    return float((x - ref).norm() / ref.norm().clamp_min(1e-30))


def make_batch(vocab, B=6, T=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, vocab, (B, T), generator=g)
    attn = torch.ones_like(ids)
    attn[1, T - 7:] = 0          # right padding
    attn[2, :5] = 0              # left padding
    attn[4, :3] = 0
    attn[4, T - 2:] = 0          # both
    loss_mask = attn.clone()
    loss_mask[:, :8] = 0         # score only the "completion" part
    w = torch.randn(B, T, generator=g) * loss_mask
    return ids, attn, loss_mask, w


def _ref_forward_loss(hf, ids, attn, loss_mask, w):
    pos = position_ids_from_mask(attn)
    logits = hf(input_ids=ids, attention_mask=attn, position_ids=pos).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
    lm = loss_mask[:, 1:].bool()
    loss = (w[:, 1:] * lp * lm).sum()
    return logits, lp, loss


def _engine_loss(eng, ids, attn, loss_mask, w, micro_batch=4, with_ref=False):
    wd = w.to(eng.device)

    def loss_fn(lp, ref_lp, b, t):
        return (wd[b, t] * lp).sum()
    return eng.train_pass(ids, attn, loss_mask, loss_fn, micro_batch=micro_batch, with_ref=with_ref)


def run_all(n_layers: int = 4, gen_prompts: int = 50, verbose: bool = True) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="tiny_qwen3_"))
    mdir = make_tiny_qwen3(tmp, n_layers=n_layers)
    res: dict = {"model": "tiny Qwen3 (random)", "n_layers": n_layers}

    # ---------------- fp32: logits + grads
    st32 = WeightStore(mdir, dtype=torch.float32)
    lora = LoRA(st32.dims, r=4, alpha=8, seed=1, b_init_std=0.05)  # B != 0 so dL/dA != 0
    eng = L2LEngine(st32, lora=lora)
    hf = load_hf(mdir, torch.float32, "eager")
    ref_params = attach_reference_lora(hf, lora)
    ids, attn, loss_mask, w = make_batch(st32.dims.vocab)
    real = attn.bool()

    ref_logits, ref_lp, ref_loss = _ref_forward_loss(hf, ids, attn, loss_mask, w)
    ref_loss.backward()
    got_logits = eng.full_logits(ids, attn, micro_batch=4)
    c0 = eng.streamer.stats.copies
    out = _engine_loss(eng, ids, attn, loss_mask, w)
    copies_train = eng.streamer.stats.copies - c0
    g_err = [nmax(p.grad, r.grad) for p, r in zip(lora.parameters(), ref_params)]
    res["fp32"] = {
        "logits_nmax": nmax(got_logits[real], ref_logits[real]),
        "logits_rfro": rfro(got_logits[real], ref_logits[real]),
        "loss_engine": out["loss"], "loss_ref": float(ref_loss),
        "grad_nmax_max": max(g_err), "grad_nmax_median": sorted(g_err)[len(g_err) // 2],
        "n_grad_tensors": len(g_err),
        "grad_rfro_max": max(rfro(p.grad, r.grad) for p, r in zip(lora.parameters(), ref_params)),
    }
    res["fp32"]["PASS"] = res["fp32"]["logits_nmax"] <= 1e-5 and res["fp32"]["grad_nmax_max"] <= 1e-5

    # ---------------- copies (H4): each layer at most once per direction per pass,
    # independent of batch size. (The layer still sitting in the ring buffer when
    # forward turns into backward may be reused, so the count can be 2L - 1.)
    def train_copies(B, mb):
        e = L2LEngine(st32, lora=lora)
        before = list(e.streamer.stats.per_layer_copies)
        _engine_loss(e, ids[:B], attn[:B], loss_mask[:B], w[:B], micro_batch=mb)
        per = [a - b for a, b in zip(e.streamer.stats.per_layer_copies, before)]
        return sum(per), max(per)
    small, big = train_copies(2, 1), train_copies(6, 2)
    res["copies"] = {"train_pass_B2_mb1": small[0], "train_pass_B6_mb2": big[0],
                     "max_copies_of_one_layer": max(small[1], big[1]),
                     "bounds": [2 * n_layers - 1, 2 * n_layers],
                     "PASS": small == big and max(small[1], big[1]) <= 2
                     and 2 * n_layers - 1 <= big[0] <= 2 * n_layers}
    for p in lora.parameters():
        p.grad = None
    _engine_loss(eng, ids, attn, loss_mask, w)  # restore grads of the main engine run
    del copies_train

    # ---------------- KL reference = adapter off, same pass
    for p in lora.parameters():
        p.grad = None
    out_ref = _engine_loss(eng, ids, attn, loss_mask, w, with_ref=True)
    base_lp = eng.logprobs(ids, attn, loss_mask, lora_enabled=False)
    m = out_ref["ref_logp"] != 0
    res["ref_logp_vs_adapter_off"] = nmax(out_ref["ref_logp"][m], base_lp[m])

    # ---------------- padding invariance
    seq = ids[0, :16]
    lp_nopad = eng.logprobs(seq[None], torch.ones(1, 16, dtype=torch.long), torch.ones(1, 16, dtype=torch.long))
    ids_l = torch.cat([torch.zeros(5, dtype=torch.long), seq])[None]
    attn_l = torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(16, dtype=torch.long)])[None]
    lp_left = eng.logprobs(ids_l, attn_l, attn_l)[:, 5:]
    ids_r = torch.cat([seq, torch.zeros(5, dtype=torch.long)])[None]
    attn_r = torch.cat([torch.ones(16, dtype=torch.long), torch.zeros(5, dtype=torch.long)])[None]
    lp_right = eng.logprobs(ids_r, attn_r, attn_r)[:, :16]
    res["padding_invariance_nmax"] = max(nmax(lp_left, lp_nopad), nmax(lp_right, lp_nopad))

    # ---------------- buffer / residency variants give identical results
    variants = {}
    for name, kw in {"n_buffers=2": dict(n_buffers=2), "resident_layers=2": dict(resident_layers=2),
                     "verify_copies": dict(verify_copies=True)}.items():
        e2 = L2LEngine(st32, lora=lora, **kw)
        variants[name] = nmax(e2.full_logits(ids, attn)[real], got_logits[real])
    res["variants_nmax"] = variants

    # ---------------- generation (H18): greedy, KV cache, LoRA on, vs HF generate
    g = torch.Generator().manual_seed(3)
    prompts = [torch.randint(3, st32.dims.vocab, (int(torch.randint(4, 12, (1,), generator=g)),),
                             generator=g).tolist() for _ in range(gen_prompts)]
    max_new = 20
    c0 = eng.streamer.stats.copies
    loaded_before = {x for x in eng.streamer.slot_layer if x >= 0}
    gen = eng.generate(prompts, max_new, temperature=0.0, stop_ids=(2,), pad_id=0, batch_size=16)
    gen_copies = eng.streamer.stats.copies - c0
    P = max(len(p) for p in prompts)
    hf_ids = torch.zeros(len(prompts), P, dtype=torch.long)
    hf_attn = torch.zeros_like(hf_ids)
    for i, p in enumerate(prompts):
        hf_ids[i, P - len(p):] = torch.tensor(p)
        hf_attn[i, P - len(p):] = 1
    with torch.no_grad():
        hf_out = hf.generate(input_ids=hf_ids, attention_mask=hf_attn, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=2, pad_token_id=0)[:, P:]
    same = 0
    first_diff = None
    for i, o in enumerate(gen):
        ref_toks = hf_out[i].tolist()
        if 2 in ref_toks:
            ref_toks = ref_toks[:ref_toks.index(2) + 1]
        if o.tokens == ref_toks:
            same += 1
        elif first_diff is None:
            first_diff = {"prompt": i, "l2l": o.tokens, "hf": ref_toks}
    res["generation_greedy"] = {"prompts": len(prompts), "identical": same, "first_diff": first_diff,
                                "PASS": same == len(prompts)}
    # copies during generation: one per layer per pass (prefill + decode steps), batches of 16
    n_passes = 0
    for s in range(0, len(prompts), 16):
        chunk = gen[s:s + 16]
        n_passes += max(len(o.tokens) for o in chunk)  # prefill + (max_len - 1) decode passes
    res["copies"]["generation"] = gen_copies
    # If every layer fits in the ring (n_layers <= n_buffers), each layer is
    # copied once and then stays on the GPU; otherwise once per layer per pass.
    n_buf = len(eng.streamer.bufs)
    res["copies"]["generation_expected"] = (n_layers - len(loaded_before)) if n_layers <= n_buf \
        else n_passes * n_layers
    res["copies"]["PASS"] = res["copies"]["PASS"] and gen_copies == res["copies"]["generation_expected"]

    # ---------------- sampled log-probs = training-forward log-probs (on-policy consistency)
    gs = torch.Generator().manual_seed(5)
    samp = eng.generate(prompts[:16], 16, temperature=1.0, stop_ids=(2,), pad_id=0, batch_size=16, generator=gs)
    seqs = [p + o.tokens for p, o in zip(prompts[:16], samp)]
    L = max(len(s) for s in seqs)
    sid = torch.zeros(16, L, dtype=torch.long)
    sat = torch.zeros_like(sid)
    slm = torch.zeros_like(sid)
    for i, (p, o) in enumerate(zip(prompts[:16], samp)):
        s = p + o.tokens
        sid[i, :len(s)] = torch.tensor(s)
        sat[i, :len(s)] = 1
        slm[i, len(p):len(s)] = 1
    lp_train = eng.logprobs(sid, sat, slm)
    lp_gen = torch.zeros_like(lp_train)
    for i, (p, o) in enumerate(zip(prompts[:16], samp)):
        lp_gen[i, len(p):len(p) + len(o.tokens)] = torch.tensor(o.logprobs)
    mm = slm.bool()
    res["gen_vs_train_logp_nmax"] = nmax(lp_gen[mm], lp_train[mm])

    # ---------------- bf16 with measured noise floor
    st16 = WeightStore(mdir, dtype=torch.bfloat16)
    lora16 = LoRA(st16.dims, r=4, alpha=8, seed=1, b_init_std=0.05)
    eng16 = L2LEngine(st16, lora=lora16)
    runs = {}
    for impl in ("eager", "sdpa"):
        h = load_hf(mdir, torch.bfloat16, impl)
        rp = attach_reference_lora(h, lora16)
        lg, _, ls = _ref_forward_loss(h, ids, attn, loss_mask, w)
        ls.backward()
        runs[impl] = (lg[real].float(), [p.grad.clone() for p in rp])
    l16 = eng16.full_logits(ids, attn, micro_batch=4)[real]
    _engine_loss(eng16, ids, attn, loss_mask, w)
    g16 = [p.grad for p in lora16.parameters()]
    truth_logits = ref_logits[real]
    truth_grads = [r.grad for r in ref_params]

    def gerr(fn, a, b):
        return max(fn(x, y) for x, y in zip(a, b))

    bf = {}
    for metric, fn in (("nmax", nmax), ("rfro", rfro)):
        noise_l = fn(runs["eager"][0], runs["sdpa"][0])
        noise_g = gerr(fn, runs["eager"][1], runs["sdpa"][1])
        e_l = min(fn(l16, runs["eager"][0]), fn(l16, runs["sdpa"][0]))
        e_g = min(gerr(fn, g16, runs["eager"][1]), gerr(fn, g16, runs["sdpa"][1]))
        bf[metric] = {
            "noise_logits": noise_l, "l2l_logits": e_l,
            "noise_grads": noise_g, "l2l_grads": e_g,
            "vs_fp32_truth": {
                "l2l_logits": fn(l16, truth_logits), "eager_logits": fn(runs["eager"][0], truth_logits),
                "sdpa_logits": fn(runs["sdpa"][0], truth_logits),
                "l2l_grads": gerr(fn, g16, truth_grads), "eager_grads": gerr(fn, runs["eager"][1], truth_grads),
                "sdpa_grads": gerr(fn, runs["sdpa"][1], truth_grads),
            },
        }
    res["bf16"] = bf
    if verbose:
        import json
        print(json.dumps(res, indent=2, default=str))
    return res
