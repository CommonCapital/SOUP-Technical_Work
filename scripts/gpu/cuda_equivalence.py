"""Equivalence on CUDA with the copy stream + prefetch ON (silent-failure row
"prefetch race: layer N computes with layer N+1's weights").

Chain of evidence: CPU engine == HF transformers (scripts/equivalence_test.py);
this script shows CUDA engine (prefetch, 2 and 3 buffers, verify_copies) ==
CPU engine in fp32 on logits, LoRA gradients and greedy generation, on a model
with more layers than buffers so the ring wraps many times.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.equivalence import _engine_loss, make_batch, nmax  # noqa: E402
from soupgrpo.l2l import L2LEngine  # noqa: E402
from soupgrpo.lora import LoRA  # noqa: E402
from soupgrpo.tiny import make_tiny_qwen3  # noqa: E402
from soupgrpo.weights import WeightStore  # noqa: E402


def run(store, device, **kw):
    lora = LoRA(store.dims, r=4, alpha=8, seed=1, b_init_std=0.05, device=device)
    eng = L2LEngine(store, device=device, lora=lora, **kw)
    ids, attn, lm, w = make_batch(store.dims.vocab, B=8, T=32)
    logits = eng.full_logits(ids, attn, micro_batch=3)
    _engine_loss(eng, ids, attn, lm, w, micro_batch=3)
    grads = [p.grad.detach().cpu() for p in lora.parameters()]
    prompts = [ids[i, :10].tolist() for i in range(8)]
    gen = [o.tokens for o in eng.generate(prompts, 24, 0.0, (2,), 0, 8)]
    return logits, grads, gen, attn.bool()


def main():
    mdir = make_tiny_qwen3(tempfile.mkdtemp(), n_layers=12)
    store = WeightStore(mdir, dtype=torch.float32, pin=True)
    ref_l, ref_g, ref_gen, real = run(store, "cpu")
    res = {}
    for name, kw in {"prefetch_3buf": dict(n_buffers=3), "prefetch_2buf": dict(n_buffers=2),
                     "prefetch_3buf_verify": dict(n_buffers=3, verify_copies=True),
                     "no_prefetch": dict(prefetch=False)}.items():
        l, g, gen, _ = run(store, "cuda", **kw)
        res[name] = {"logits_nmax": nmax(l[real], ref_l[real]),
                     "grad_nmax_max": max(nmax(a, b) for a, b in zip(g, ref_g)),
                     "greedy_identical": gen == ref_gen}
        res[name]["PASS"] = (res[name]["logits_nmax"] <= 1e-5 and res[name]["grad_nmax_max"] <= 1e-5
                             and res[name]["greedy_identical"])
    res["PASS"] = all(v["PASS"] for v in res.values())
    out = ROOT / "results" / "logs" / f"cuda_equivalence_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["PASS"] else 1)


if __name__ == "__main__":
    main()
