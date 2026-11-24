"""Pilot and evaluation: run a model (base, base+adapter, or merged) on a
puzzle file and score every answer.

Pilot (plan step 8):   --samples 4 --temperature 1.0  on data/pilot_<level>.jsonl
Evaluation (step 12):  --samples 1 --temperature 0     on data/heldout_<level>.jsonl

Writes <out>.jsonl (one row per answer, full text) and <out>.summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soupgrpo.runtime import (build_engine, encode_prompts, gpu_mem, load_puzzles,  # noqa: E402
                              nvidia_smi_used_mib, score_completions, stop_ids, tokenizer_for)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"], help="prompt version (prompts.py)")
    a = ap.parse_args()

    eng, mdir = build_engine(a.model, a.device, a.dtype, adapter=a.adapter)
    tok = tokenizer_for(mdir)
    stops = stop_ids(tok)
    puzzles = load_puzzles(a.data, a.limit)
    pids = encode_prompts(tok, puzzles, a.prompt)
    rep_p = [p for p in pids for _ in range(a.samples)]
    rep_z = [z for z in puzzles for _ in range(a.samples)]
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    t0 = time.perf_counter()
    outs = eng.generate(rep_p, a.max_new, a.temperature, stops, tok.pad_token_id, a.batch, gen)
    secs = time.perf_counter() - t0
    rows = score_completions(tok, rep_z, outs, stops)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".jsonl"), "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"sample": i % a.samples, **r}) + "\n")

    rew = np.array([r["reward"] for r in rows]).reshape(len(puzzles), a.samples)
    per_prompt = rew.mean(1)
    k, n = int(rew.sum()), rew.size
    lo, hi = wilson(k, n)
    summ = {
        "model": a.model, "adapter": a.adapter, "data": a.data, "n_prompts": len(puzzles), "samples": a.samples,
        "temperature": a.temperature, "max_new": a.max_new,
        "accuracy": k / n, "wilson95": [lo, hi],
        "mixed_prompt_share": float(np.mean((per_prompt > 0) & (per_prompt < 1))) if a.samples > 1 else None,
        "all_wrong_share": float(np.mean(per_prompt == 0)), "all_right_share": float(np.mean(per_prompt == 1)),
        "per_prompt_hist": {str(v): int((rew.sum(1) == v).sum()) for v in range(a.samples + 1)},
        "tag_rate": float(np.mean([r["tag_present"] for r in rows])),
        "parsed_rate": float(np.mean([r["parsed"] for r in rows])),
        "numbers_rate": float(np.mean([r["numbers_correct"] for r in rows])),
        "value_rate": float(np.mean([r["value_correct"] for r in rows])),
        "truncated_share": float(np.mean([r["finish"] == "length" for r in rows])),
        "thinking_share": float(np.mean([r["thinking"] for r in rows])),
        "len_mean": float(np.mean([r["n_tokens"] for r in rows])),
        "len_p95": float(np.percentile([r["n_tokens"] for r in rows], 95)),
        "seconds": secs, "engine_timers_s": dict(eng.timers), "copies": eng.streamer.stats.copies,
        "dtype": a.dtype, "prompt": a.prompt, "nonfinite_events": eng.nonfinite_events,
        "gpu": gpu_mem(), "smi_mib": nvidia_smi_used_mib(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps({k: summ[k] for k in ("accuracy", "wilson95", "mixed_prompt_share", "per_prompt_hist",
                                           "tag_rate", "value_rate", "truncated_share", "thinking_share",
                                           "len_mean", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
