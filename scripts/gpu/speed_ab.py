"""Speed A/B tests of the L2L engine on the real GPU (pre-registered in
predictions/PREDICTIONS.md, "Speed A/B and DFlash"). One process, one GPU, so
all arms share the host, the PCIe link and the thermal state; arms alternate
A-B-B-A (twice) so drift (clocks, temperature) hits both arms equally.

H3  prefetch on vs off, in decoding and in training. Hidden share of copy time
    = (t_off - t_on) / t_copy, t_copy = 36 measured layer copies.
H2  decoding throughput at EQUAL memory: 0 resident layers + batch 48 vs
    8 resident layers + the batch that fits in the same memory.
H5  per-phase placement (the 10Cache / D2 idea): training with 16 layers
    resident (loaded before the phase, freed after, both timed) vs the fixed
    plan (0 resident everywhere). Generation is the same in both plans, so
    only the training phase is timed.

  python scripts/gpu/speed_ab.py [--quick]      -> results/logs/speed_ab_<ts>.json
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
from soupgrpo.grpo import make_loss_fn, pack_sequences  # noqa: E402
from soupgrpo.runtime import build_engine, encode_prompts, load_puzzles, tokenizer_for  # noqa: E402

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"


def sync():
    if CUDA:
        torch.cuda.synchronize()


def peak_reset():
    if CUDA:
        torch.cuda.reset_peak_memory_stats()


def peak_gib():
    return torch.cuda.max_memory_allocated() / 2**30 if CUDA else 0.0


def copy_time_per_layer(eng, reps=20):
    st = eng.store
    buf = eng.streamer.bufs[0]
    sync()
    t = time.perf_counter()
    for i in range(reps):
        buf.copy_(st.layers[i % st.dims.n_layers], non_blocking=True)
    sync()
    return (time.perf_counter() - t) / reps


def decode_run(eng, pids, max_new, pad):
    eng.timers.clear()
    peak_reset()
    outs = eng.generate(pids, max_new, 0.0, stop_ids=(), pad_id=pad, batch_size=len(pids))
    return {"pass_s": eng.timers["decode"] / (max_new - 1), "prefill_s": eng.timers["prefill"],
            "tokens_per_s": len(pids) / (eng.timers["decode"] / (max_new - 1)), "peak_gib": peak_gib(),
            "tokens": [o.tokens for o in outs]}


def train_phase(eng, packs, resident=0):
    """One training phase as in a real step: 2 passes of 24 sequences + Adam step.
    resident > 0: load that many layers first and free them after (both timed)."""
    opt = torch.optim.AdamW(eng.lora.parameters(), lr=0.0)
    sync()
    peak_reset()
    t0 = time.perf_counter()
    if resident:
        eng.streamer.set_resident(resident)
    t_load = time.perf_counter() - t0
    for ids, attn, lm, adv in packs:
        eng.train_pass(ids, attn, lm, make_loss_fn(adv, 0.0, float(lm.sum()), {}), 8)
    opt.step()
    opt.zero_grad(set_to_none=True)
    sync()
    peak = peak_gib()
    t1 = time.perf_counter()
    if resident:
        eng.streamer.set_resident(0)
    t_free = time.perf_counter() - t1
    return {"seconds": time.perf_counter() - t0, "load_s": t_load, "free_s": t_free, "peak_gib": peak}


def abba(fn_a, fn_b, rounds):
    res = {"A": [], "B": [], "order": []}
    for _ in range(rounds):
        for arm in ("A", "B", "B", "A"):
            res[arm].append((fn_a if arm == "A" else fn_b)())
            res["order"].append(arm)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--data", default="data/train_n5_main.jsonl")
    ap.add_argument("--quick", action="store_true", help="tiny sizes (local CPU test of the script)")
    a = ap.parse_args()
    q = a.quick
    B0, DEC_NEW, KV_LEN, ANS_LEN, ROUNDS = (4, 6, 64, 20, 1) if q else (48, 64, 512 + 200, 450, 2)
    R_GEN, R_TRAIN = (1, 2) if q else (8, 16)

    eng, mdir = build_engine(a.model, DEV, a.dtype, lora_r=16, lora_alpha=32)
    tok = tokenizer_for(mdir)
    pad = tok.pad_token_id
    pids = encode_prompts(tok, load_puzzles(ROOT / a.data, B0), "v2")
    d = eng.d
    res = {"config": {**vars(a), "B0": B0, "decode_new": DEC_NEW, "kv_len": KV_LEN, "answer_len": ANS_LEN,
                      "rounds": ROUNDS, "R_gen": R_GEN, "R_train": R_TRAIN},
           "device": torch.cuda.get_device_name(0) if CUDA else "cpu"}
    res["layer_mb"] = eng.store.layer_bytes / 1e6
    res["copy_s_per_layer"] = copy_time_per_layer(eng)
    res["t_copy_pass_s"] = res["copy_s_per_layer"] * d.n_layers
    print("layer copy", round(res["copy_s_per_layer"] * 1e3, 2), "ms; 36 layers", round(res["t_copy_pass_s"], 3), "s",
          flush=True)
    eng.kv_min_len = KV_LEN       # KV cache sized as in real rollouts (prompt + 512 answer tokens)

    # training inputs with the real shape: 2 passes x 24 sequences x (prompt + ~450 answer tokens).
    # Timing does not depend on the token values, so answers are random tokens.
    g = torch.Generator().manual_seed(0)
    n_pass = B0 // 2
    packs = []
    for s in (0, n_pass):
        comp = [torch.randint(10, 150000 if not q else 400, (ANS_LEN,), generator=g).tolist() for _ in range(n_pass)]
        ids, attn, lm = pack_sequences(pids[s:s + n_pass], comp, pad)
        packs.append((ids, attn, lm, torch.randn(n_pass, generator=g).to(DEV)))

    # host buffer for per-layer inputs: reserve the largest pass once (E14: growing it ran the host out of RAM)
    eng.boundaries.reserve(d.n_layers, max(p[0].numel() for p in packs) * d.hidden, eng.res_dtype)

    # ---- warm-up (cuBLAS handles, allocator)
    decode_run(eng, pids[:4], 4, pad)
    train_phase(eng, packs[:1])

    # ---- H3: prefetch on (A) vs off (B)
    def dec(on):
        eng.streamer.set_prefetch(on)
        r = decode_run(eng, pids, DEC_NEW, pad)
        r.pop("tokens")
        return r

    def trn(on):
        eng.streamer.set_prefetch(on)
        return train_phase(eng, packs)
    res["H3_decode"] = abba(lambda: dec(True), lambda: dec(False), ROUNDS)
    res["H3_train"] = abba(lambda: trn(True), lambda: trn(False), ROUNDS)
    eng.streamer.set_prefetch(True)
    for ph, key in (("decode", "pass_s"), ("train", "seconds")):
        r = res[f"H3_{ph}"]
        on, off = np.mean([x[key] for x in r["A"]]), np.mean([x[key] for x in r["B"]])
        copies = (1 if ph == "decode" else 2 * 2) * res["t_copy_pass_s"]   # train: 2 passes x (fwd + bwd)
        r["mean_on"], r["mean_off"], r["t_copy"] = on, off, copies
        r["hidden_share"] = (off - on) / copies
        print(f"H3 {ph}: on {on:.3f}s off {off:.3f}s copy {copies:.3f}s -> hidden {r['hidden_share']:.0%}", flush=True)

    # ---- H2: equal memory. 8 resident layers cost 8 x layer bytes = this many sequences of KV:
    kv_per_seq = d.n_layers * 2 * d.n_kv_heads * KV_LEN * d.head_dim * eng.store.layer_bytes // eng.store.layer_numel
    B8 = max(1, B0 - int(np.ceil(R_GEN * eng.store.layer_bytes / kv_per_seq)))
    res["H2_kv_bytes_per_seq"], res["H2_B8"] = kv_per_seq, B8

    def gen_r(R, B):
        eng.streamer.set_resident(R)
        r = decode_run(eng, pids[:B], DEC_NEW, pad)
        eng.streamer.set_resident(0)
        return r
    h2 = abba(lambda: gen_r(0, B0), lambda: gen_r(R_GEN, B8), ROUNDS)
    same = all(h2["A"][0]["tokens"][i] == h2["B"][0]["tokens"][i] for i in range(B8))
    for arm in ("A", "B"):
        for x in h2[arm]:
            x.pop("tokens")
    h2["greedy_tokens_identical_R0_vs_R8"] = same
    for arm, B in (("A", B0), ("B", B8)):
        h2[f"tokens_per_s_{arm}"] = float(np.mean([x["tokens_per_s"] for x in h2[arm]]))
        h2[f"peak_gib_{arm}"] = float(max(x["peak_gib"] for x in h2[arm]))
    h2["ratio_R0_over_R8"] = h2["tokens_per_s_A"] / h2["tokens_per_s_B"]
    res["H2"] = h2
    print(f"H2: R=0 B={B0} {h2['tokens_per_s_A']:.1f} tok/s ({h2['peak_gib_A']:.2f} GiB) vs R={R_GEN} B={B8} "
          f"{h2['tokens_per_s_B']:.1f} tok/s ({h2['peak_gib_B']:.2f} GiB); identical tokens {same}", flush=True)

    # ---- H5: fixed plan (0 resident) vs per-phase plan (16 resident during training)
    h5 = abba(lambda: train_phase(eng, packs, 0), lambda: train_phase(eng, packs, R_TRAIN), ROUNDS)
    h5["mean_fixed"] = float(np.mean([x["seconds"] for x in h5["A"]]))
    h5["mean_per_phase"] = float(np.mean([x["seconds"] for x in h5["B"]]))
    h5["saving_s"] = h5["mean_fixed"] - h5["mean_per_phase"]
    h5["peak_gib_per_phase"] = float(max(x["peak_gib"] for x in h5["B"]))
    res["H5"] = h5
    print(f"H5: fixed {h5['mean_fixed']:.2f}s vs per-phase {h5['mean_per_phase']:.2f}s "
          f"(saving {h5['saving_s']:+.2f}s per step; peak {h5['peak_gib_per_phase']:.2f} GiB)", flush=True)

    out = ROOT / "results" / "logs" / f"speed_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float) + "\n")
    print("log:", out)


if __name__ == "__main__":
    main()
