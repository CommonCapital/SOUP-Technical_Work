"""H30: fp16 (T4 has no bf16) does not change Qwen3-4B's behaviour.

Reference = fp32 compute on the released bf16 weights (exact upcast), streamed
by the same L2L engine. On N pilot prompts: the fp32 model writes greedy
answers; then both models score those exact sequences (teacher forcing).
Reports per-position top-1 agreement, |Δ log p| of the chosen tokens, max
|Δ logit|, non-finite events, and fp16's own greedy answers vs fp32's.

Decision rule (pre-registered, H30): PASS if no non-finite values, top-1
agreement >= 99% and mean |Δ log p| <= 0.01 nats.
Each model runs in its own subprocess so CPU RAM never holds both stores.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def worker(model, device, dtype, n, max_new, data, out, ref_seqs=None):
    from soupgrpo.grpo import pack_sequences
    from soupgrpo.runtime import build_engine, encode_prompts, load_puzzles, stop_ids, tokenizer_for
    eng, mdir = build_engine(model, device, dtype)
    tok = tokenizer_for(mdir)
    pids = encode_prompts(tok, load_puzzles(data, n))
    t0 = time.perf_counter()
    gen = [o.tokens for o in eng.generate(pids, max_new, 0.0, stop_ids(tok), tok.pad_token_id, 64)]
    gen_s = time.perf_counter() - t0
    seqs = ref_seqs or gen
    ids, attn, lm = pack_sequences(pids, seqs, tok.pad_token_id)
    logits = eng.full_logits(ids, attn, micro_batch=2)            # [B, T, V] fp32 on CPU
    m = lm.clone()
    m[:, 0] = 0
    b, t = m.nonzero(as_tuple=True)
    pred = logits[b, t - 1]
    res = {"argmax": pred.argmax(-1), "logp": torch.log_softmax(pred, -1).gather(-1, ids[b, t][:, None])[:, 0],
           "logits": pred.half(), "gen": gen, "gen_seconds": gen_s,
           "nonfinite": eng.nonfinite_events + int((~torch.isfinite(pred)).sum())}
    torch.save(res, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--low", default="fp16", help="dtype under test")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--data", default=str(ROOT / "data" / "pilot_n4.jsonl"))
    ap.add_argument("--worker", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    if a.worker:
        ref = torch.load(a.ref)["gen"] if a.ref else None
        worker(a.model, a.device, a.worker, a.n, a.max_new, a.data, a.out, ref)
        return

    tmp = ROOT / "results" / "logs"
    f32, f16 = tmp / "_fp16check_fp32.pt", tmp / "_fp16check_fp16.pt"
    base = [sys.executable, __file__, "--n", str(a.n), "--max-new", str(a.max_new), "--data", a.data,
            "--model", a.model, "--device", a.device]
    subprocess.run(base + ["--worker", "fp32", "--out", str(f32)], check=True)
    subprocess.run(base + ["--worker", a.low, "--out", str(f16), "--ref", str(f32)], check=True)
    r32, r16 = torch.load(f32), torch.load(f16)
    dlp = (r16["logp"] - r32["logp"]).abs()
    tok_agree = tok_total = exact = 0
    for x, y in zip(r32["gen"], r16["gen"]):
        k = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
        tok_agree += k
        tok_total += max(len(x), len(y))
        exact += x == y
    res = {
        "hypothesis": "H30", "model": a.model, "dtype_under_test": a.low, "n_prompts": a.n, "scored_positions": int(len(dlp)),
        "top1_agreement_teacher_forced": float((r16["argmax"] == r32["argmax"]).float().mean()),
        "mean_abs_dlogp": float(dlp.mean()), "max_abs_dlogp": float(dlp.max()),
        "max_abs_dlogit": float((r16["logits"].float() - r32["logits"].float()).abs().max()),
        "greedy_exact_match_share": exact / a.n, "greedy_token_agreement_until_first_diff": tok_agree / tok_total,
        "nonfinite_fp16": r16["nonfinite"], "nonfinite_fp32": r32["nonfinite"],
        "gen_seconds": {"fp32": r32["gen_seconds"], "fp16": r16["gen_seconds"]},
    }
    res["PASS"] = (res["nonfinite_fp16"] == 0 and res["top1_agreement_teacher_forced"] >= 0.99
                   and res["mean_abs_dlogp"] <= 0.01)
    out = tmp / f"fp16_check_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    f32.unlink(missing_ok=True)
    f16.unlink(missing_ok=True)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
