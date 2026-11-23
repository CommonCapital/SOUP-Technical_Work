"""Plan step 7: memory and time budget on the real GPU (H11, H12, H1, H15, H4).

1. Load Qwen3-4B into the L2L engine (LoRA r16). Record nvidia-smi + allocator.
2. One generation batch: 64 prompts, forced to run the full 256 new tokens
   (no stop tokens), so the KV cache reaches its planned size.
3. One training pass over those 64 sequences.
4. H12: logits unchunked (chunk = 8192 rows) vs 512-row chunks on 8k tokens.
Every phase records allocated / reserved / nvidia-smi peaks and time, next to
the paper estimate. Writes results/logs/budget_<ts>.json.
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.grpo import make_loss_fn, pack_sequences  # noqa: E402
from soupgrpo.runtime import build_engine, encode_prompts, load_puzzles, tokenizer_for  # noqa: E402

ESTIMATE_GB = {"rollout_total": 6.7, "context": 0.5, "embedding": 0.78, "layer_buffers": 0.6, "lora_adam": 0.53,
               "kv_64x400": 3.7}


class SmiPeak:
    """Polls nvidia-smi every 100 ms in a thread; .peak() since last reset."""

    def __init__(self):
        self._peak, self._stop = 0, False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            try:
                v = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5).stdout.split()[0])
                self._peak = max(self._peak, v)
            except Exception:
                pass
            time.sleep(0.1)

    def reset(self):
        p, self._peak = self._peak, 0
        return p


def phase(name, res, smi, t0):
    torch.cuda.synchronize()
    time.sleep(0.3)
    res[name] = {"seconds": time.perf_counter() - t0,
                 "alloc_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "reserved_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
                 "smi_peak_mib": smi.reset()}
    print(name, json.dumps(res[name]), flush=True)
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=8)
    a = ap.parse_args()

    smi = SmiPeak()
    res = {"estimate_gb": ESTIMATE_GB}
    t = time.perf_counter()
    eng, mdir = build_engine(a.model, "cuda", "bf16", lora_r=16, lora_alpha=32)
    t = phase("load", res, smi, t)
    res["lora_params"] = eng.lora.n_params()
    tok = tokenizer_for(mdir)

    puzzles = load_puzzles(ROOT / "data" / "pilot_n4.jsonl", a.batch)
    pids = encode_prompts(tok, puzzles)
    eng.timers.clear()
    c0 = eng.streamer.stats.copies
    outs = eng.generate(pids, a.max_new, 1.0, stop_ids=(), pad_id=tok.pad_token_id, batch_size=a.batch,
                        generator=torch.Generator(device="cuda").manual_seed(0))
    res["gen_copies"] = eng.streamer.stats.copies - c0
    res["gen_timers"] = dict(eng.timers)
    res["decode_pass_s"] = eng.timers["decode"] / (a.max_new - 1)
    t = phase("generate", res, smi, t)

    ids, attn, lm = pack_sequences(pids, [o.tokens for o in outs], tok.pad_token_id)
    adv = torch.randn(len(outs), device="cuda")
    eng.timers.clear()
    eng.train_pass(ids, attn, lm, make_loss_fn(adv, 0.0, float(len(outs) * a.max_new), {}), a.micro_batch)
    res["train_timers"] = dict(eng.timers)
    res["train_tokens"] = int(attn.sum())
    t = phase("train_pass", res, smi, t)

    # H12: unchunked logits on ~8k scored tokens
    n_seq = max(1, 8192 // a.max_new)
    sl = slice(0, n_seq)
    for chunk in (512, 8192):
        eng.logit_chunk = chunk
        try:
            eng.train_pass(ids[sl], attn[sl], lm[sl], make_loss_fn(adv[sl], 0.0, 1.0, {}), a.micro_batch)
            ok = True
        except torch.OutOfMemoryError:
            ok = False
            torch.cuda.empty_cache()
        t = phase(f"logits_chunk_{chunk}", res, smi, t)
        res[f"logits_chunk_{chunk}"]["fits"] = ok
    eng.logit_chunk = 512

    res["max_smi_mib_all_phases"] = max(v["smi_peak_mib"] for v in res.values() if isinstance(v, dict) and "smi_peak_mib" in v)
    res["under_8192_mib"] = res["max_smi_mib_all_phases"] < 8192
    out = ROOT / "results" / "logs" / f"budget_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=2, default=float) + "\n")
    print("log:", out)


if __name__ == "__main__":
    main()
