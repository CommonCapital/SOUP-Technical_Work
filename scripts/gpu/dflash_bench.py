"""DFlash on the real GPU (pre-registered in predictions/PREDICTIONS.md,
"Speed A/B and DFlash"). Subcommands:

  greedy   H21 + H7: DFlash greedy on held-out prompts with a given adapter
           (or none). Compared token by token with plain L2L greedy (run here
           with --plain, and/or the saved evaluation outputs with --ref). Every
           divergence is checked: is it a floating-point near-tie in the plain
           run (top-2 logit gap < 0.05)? Also: acceptance length, passes, time.
  sweep    H23: plain vs DFlash throughput at several batch sizes, T = 1.0,
           512 new tokens (the rollout setting).
  sampling H22: one prompt, N samples of 16 tokens at T = 1.0, plain vs DFlash;
           chi-square per position 1..15. Power check: the deliberately biased
           'accept_all' rule must be detected by the same test.

  python scripts/gpu/dflash_bench.py greedy --drafter DIR [--adapter DIR] [--plain] [--ref saved.jsonl] --tag main
Writes results/logs/dflash_<cmd>_<tag>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.dflash import DFlashGenerator, Drafter  # noqa: E402
from soupgrpo.runtime import build_engine, encode_prompts, load_puzzles, stop_ids, tokenizer_for  # noqa: E402
from soupgrpo.stats import chi2_homogeneity  # noqa: E402

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"
NEAR_TIE = 0.05


def sync():
    if CUDA:
        torch.cuda.synchronize()


def timed(fn):
    sync()
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t, (torch.cuda.max_memory_allocated() / 2**30 if CUDA else 0.0)


def setup(a):
    eng, mdir = build_engine(a.model, DEV, a.dtype, adapter=a.adapter)
    tok = tokenizer_for(mdir)
    dr = Drafter.load(a.drafter, DEV, eng.dtype) if not a.random_drafter else \
        Drafter.random(eng.d, 2, [1, eng.d.n_layers - 2], 16, 7, DEV, eng.dtype)
    info = {"drafter_gib": dr.nbytes() / 2**30, "drafter_max_abs_weight": dr.max_abs_weight,
            "target_layer_ids": dr.target_layer_ids, "block": dr.block_size, "mask_token_id": dr.mask_token_id,
            "device": torch.cuda.get_device_name(0) if CUDA else "cpu", "config": vars(a)}
    print("drafter:", {k: v for k, v in info.items() if k != "config"}, flush=True)
    return eng, tok, dr, info


def first_diff(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def near_tie(eng, prompt, prefix, dflash_tok):
    """Top-2 logit gap of the plain model at the first divergent position."""
    ids = torch.tensor([prompt + prefix])
    lg = eng.full_logits(ids, torch.ones_like(ids), micro_batch=1)[0, -1]
    top = torch.topk(lg, 2)
    return {"gap": float(top.values[0] - top.values[1]), "plain_top1": int(top.indices[0]),
            "dflash_token": int(dflash_tok), "dflash_is_top2": int(top.indices[1]) == int(dflash_tok),
            "gap_to_dflash_token": float(top.values[0] - lg[int(dflash_tok)])}


def cmd_greedy(a):
    eng, tok, dr, info = setup(a)
    stops = stop_ids(tok)
    puzzles = load_puzzles(ROOT / a.data, a.n)
    pids = encode_prompts(tok, puzzles, "v2")
    spec = DFlashGenerator(eng, dr)
    outs, secs, peak = timed(lambda: spec.generate(pids, a.max_new, 0.0, stops, tok.pad_token_id, a.batch))
    res = {**info, "n": len(pids), "dflash": {**spec.stats.summary(), "seconds": secs, "peak_gib": peak,
                                              "oom_splits": getattr(spec, "oom_splits", 0),
                                              "tokens": sum(len(o.tokens) for o in outs)},
           "nonfinite_events": eng.nonfinite_events}
    refs = {}
    if a.plain:
        eng.timers.clear()
        c0 = eng.streamer.stats.copies
        plain, psecs, ppeak = timed(lambda: eng.generate(pids, a.max_new, 0.0, stops, tok.pad_token_id, a.batch))
        refs["plain_here"] = [o.tokens for o in plain]
        res["plain"] = {"seconds": psecs, "peak_gib": ppeak, "passes": (eng.streamer.stats.copies - c0) // eng.d.n_layers,
                        "tokens": sum(len(o.tokens) for o in plain)}
        res["speedup_wall"] = psecs / secs
    if a.ref:
        saved = {json.loads(line)["id"]: json.loads(line)["tokens"] for line in open(ROOT / a.ref)}
        refs["saved_eval"] = [saved[p["id"]] for p in puzzles]
    for name, ref in refs.items():
        div = []
        for i, (o, r) in enumerate(zip(outs, ref)):
            k = first_diff(o.tokens, r)
            if k is not None:
                d = {"i": i, "id": puzzles[i]["id"], "pos": k}
                if k < min(len(o.tokens), len(r)):
                    d.update(near_tie(eng, pids[i], r[:k], o.tokens[k]))
                div.append(d)
        ties = [d for d in div if d.get("gap", 1e9) < NEAR_TIE]
        res[f"vs_{name}"] = {"identical": len(outs) - len(div), "n": len(outs),
                             "identical_share": 1 - len(div) / len(outs), "divergences": div,
                             "all_divergences_near_ties": len(ties) == len(div)}
        print(f"vs {name}: identical {len(outs) - len(div)}/{len(outs)}; divergences at near-ties "
              f"{len(ties)}/{len(div)}", flush=True)
    print("DFlash:", json.dumps({k: v for k, v in res["dflash"].items() if k != "accept_len_hist"}), flush=True)
    if "plain" in res:
        print(f"plain {res['plain']['seconds']:.0f}s vs DFlash {secs:.0f}s -> x{res['speedup_wall']:.2f}", flush=True)
    write(res, f"greedy_{a.tag}")


def cmd_sweep(a):
    eng, tok, dr, info = setup(a)
    stops = stop_ids(tok)
    pids = encode_prompts(tok, load_puzzles(ROOT / a.data, max(a.batches)), "v2")
    res = {**info, "batches": {}}
    for B in a.batches:
        g = torch.Generator(device=DEV)
        r = {}
        eng.timers.clear()
        dr.to("cpu")             # E19: the plain arm gets the whole budget (drafter off the GPU)
        eng.oom_splits = 0
        po, ps, pp = timed(lambda: eng.generate(pids[:B], a.max_new, 1.0, stops, tok.pad_token_id, B, g.manual_seed(0)))
        plain_splits = eng.oom_splits
        dr.to(DEV)
        spec = DFlashGenerator(eng, dr)
        try:
            so, ss, sp = timed(lambda: spec.generate(pids[:B], a.max_new, 1.0, stops, tok.pad_token_id, B,
                                                     g.manual_seed(0)))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            res["batches"][B] = {"dflash": "OOM"}
            print(f"B={B}: DFlash OOM", flush=True)
            continue
        pt, st = sum(len(o.tokens) for o in po), sum(len(o.tokens) for o in so)
        r = {"plain": {"seconds": ps, "tokens": pt, "tok_per_s": pt / ps, "peak_gib": pp, "oom_splits": plain_splits},
             "dflash": {"seconds": ss, "tokens": st, "tok_per_s": st / ss, "peak_gib": sp,
                        "oom_splits": getattr(spec, "oom_splits", 0), **spec.stats.summary()}}
        r["speedup_tok_per_s"] = r["dflash"]["tok_per_s"] / r["plain"]["tok_per_s"]
        res["batches"][B] = r
        print(f"B={B}: plain {pt / ps:.1f} tok/s, DFlash {st / ss:.1f} tok/s (x{r['speedup_tok_per_s']:.2f}), "
              f"accept {r['dflash']['mean_accept_len']:.2f}, passes {r['dflash']['passes']}, peak {sp:.2f} GiB",
              flush=True)
    write(res, f"sweep_{a.tag}")


def cmd_sampling(a):
    eng, tok, dr, info = setup(a)
    pz = load_puzzles(ROOT / a.data, 1)
    p = encode_prompts(tok, pz, "v2")[0]
    P = [p] * a.samples
    L = 16
    runs = {}
    plain, ts, _ = timed(lambda: eng.generate(P, L, 1.0, (), tok.pad_token_id, a.batch,
                                              torch.Generator(device=DEV).manual_seed(11)))
    runs["plain"] = (plain, ts, None)
    for mode, seed in (("exact", 12), ("accept_all", 13)):
        spec = DFlashGenerator(eng, dr, mode=mode)
        o, t, _ = timed(lambda: spec.generate(P, L, 1.0, (), tok.pad_token_id, a.dbatch,
                                              torch.Generator(device=DEV).manual_seed(seed)))
        runs[mode] = (o, t, spec.stats.summary())
    res = {**info, "prompt_id": pz[0]["id"], "samples": a.samples, "positions": list(range(1, L)), "tests": {}}
    for mode in ("exact", "accept_all"):
        ps = []
        for pos in range(1, L):
            ps.append(chi2_homogeneity([o.tokens[pos] for o in runs["plain"][0]],
                                       [o.tokens[pos] for o in runs[mode][0]]))
        res["tests"][mode] = {"per_position": ps, "min_p": min(x["p"] for x in ps),
                              "seconds": runs[mode][1], "stats": runs[mode][2]}
        print(f"{mode}: min p over positions 1-15 = {res['tests'][mode]['min_p']:.3g}", flush=True)
    res["plain_seconds"] = runs["plain"][1]
    alpha = 0.05 / (L - 1)
    res["PASS_exact"] = res["tests"]["exact"]["min_p"] > alpha
    res["PASS_power"] = res["tests"]["accept_all"]["min_p"] < alpha
    print("H22 PASS (exact rule not rejected):", res["PASS_exact"], "| test has power (biased rule rejected):",
          res["PASS_power"], flush=True)
    write(res, f"sampling_{a.tag}")


def write(res, name):
    out = ROOT / "results" / "logs" / f"dflash_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("log:", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["greedy", "sweep", "sampling"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--drafter", default=None)
    ap.add_argument("--random-drafter", action="store_true", help="local tests only")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/heldout_n5.jsonl")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dbatch", type=int, default=32, help="DFlash batch in 'sampling'")
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32, 48])
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    {"greedy": cmd_greedy, "sweep": cmd_sweep, "sampling": cmd_sampling}[a.cmd](a)


if __name__ == "__main__":
    main()
