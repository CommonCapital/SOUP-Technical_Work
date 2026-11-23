"""H27: merged model == base + adapter.

On N held-out prompts: greedy generation from both (token agreement), and
log-probs of the adapter's own greedy answers under both (max abs diff).
Noise floor: the adapter model run twice with different micro-batch sizes
(different kernel shapes, same math) in bf16.
Also asserts merged weights != base weights (silent failure "evaluation loads
the base path instead of the merged one").
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.grpo import pack_sequences  # noqa: E402
from soupgrpo.runtime import (build_engine, encode_prompts, load_puzzles, resolve_model_dir,  # noqa: E402
                              stop_ids, tokenizer_for)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    a = ap.parse_args()

    tok = tokenizer_for(resolve_model_dir(a.model))
    stops = stop_ids(tok)
    pids = encode_prompts(tok, load_puzzles(a.data, a.n))

    def run(model, adapter):
        eng, _ = build_engine(model, a.device, a.dtype, adapter=adapter)
        outs = eng.generate(pids, a.max_new, 0.0, stops, tok.pad_token_id, 64)
        checksum = float(sum(t.float().sum() for t in eng.store.layers[:2]))
        return eng, [o.tokens for o in outs], checksum

    eng_a, gen_a, ck_a = run(a.model, a.adapter)
    ids, attn, lm = pack_sequences(pids, gen_a, tok.pad_token_id)
    lp_a = eng_a.logprobs(ids, attn, lm, micro_batch=8)
    lp_a1 = eng_a.logprobs(ids, attn, lm, micro_batch=1)
    del eng_a
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    eng_m, gen_m, ck_m = run(a.merged, None)
    lp_m = eng_m.logprobs(ids, attn, lm, micro_batch=8)
    m = lm.bool()
    tok_agree, tok_total, exact = 0, 0, 0
    for x, y in zip(gen_a, gen_m):
        n = min(len(x), len(y))
        k = next((i for i in range(n) if x[i] != y[i]), n)
        tok_agree += k
        tok_total += max(len(x), len(y))
        exact += x == y
    res = {
        "n_prompts": len(pids),
        "greedy_exact_match_share": exact / len(pids),
        "greedy_token_agreement_until_first_diff": tok_agree / tok_total,
        "logp_max_abs_diff_merged_vs_adapter": float((lp_m[m] - lp_a[m]).abs().max()),
        "logp_noise_floor_microbatch_1_vs_8": float((lp_a1[m] - lp_a[m]).abs().max()),
        "first_layers_checksum_base": ck_a, "first_layers_checksum_merged": ck_m,
        "merged_differs_from_base": ck_a != ck_m,
    }
    res["PASS"] = (res["greedy_token_agreement_until_first_diff"] >= 0.99 and res["merged_differs_from_base"]
                   and res["logp_max_abs_diff_merged_vs_adapter"] <= 3 * max(res["logp_noise_floor_microbatch_1_vs_8"], 1e-3))
    out = ROOT / "results" / "logs" / f"merge_check_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
