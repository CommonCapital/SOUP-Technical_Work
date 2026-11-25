"""H28: how far did the adapter move the policy? Exact per-position
KL(policy || base) on the adapter model's own held-out answers.

At every answer position t (prefix = prompt + answer[:t]) both distributions
are computed over the full vocabulary from one forward pass (adapter on and
adapter off share the streamed weights), and
    KL_t = sum_v p_pol(v) * (log p_pol(v) - log p_base(v)).
Reported: mean KL per answer token, mean KL summed per answer, and the mean
log-ratio of the tokens actually written. Exact per position (no sampling
estimator), conditioned on the answers the adapter model wrote.

  python scripts/kl_eval.py --adapter runs/main/final --eval-jsonl results/eval/main.jsonl --n 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.grpo import pack_sequences  # noqa: E402
from soupgrpo.runtime import build_engine, encode_prompts, tokenizer_for  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval-jsonl", required=True, help="sample_eval output of THIS adapter (has 'tokens')")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prompt", default="v2")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--group", type=int, default=24, help="sequences per forward")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    eng, mdir = build_engine(a.model, a.device, a.dtype, adapter=a.adapter)
    tok = tokenizer_for(mdir)
    rows = [json.loads(line) for line in open(a.eval_jsonl)]
    rows = [r for r in rows if r.get("sample", 0) == 0][:a.n]
    pids = encode_prompts(tok, rows, a.prompt)
    t0 = time.perf_counter()
    kl_sum, n_tok, lr_sum, per_seq = 0.0, 0, 0.0, []
    for g in range(0, len(rows), a.group):
        ids, attn, lm = pack_sequences(pids[g:g + a.group], [r["tokens"] for r in rows[g:g + a.group]],
                                       tok.pad_token_id)
        mbs = eng._prep(ids, attn, a.micro_batch)
        xs, xr, _ = eng._forward(mbs, save_boundaries=False, with_ref=True, lora_enabled=True)
        seq_kl = torch.zeros(ids.shape[0])
        for j, m in enumerate(mbs):
            s = j * a.micro_batch
            b, t = eng._scored_index(m["attn"], lm[s:s + a.micro_batch].to(eng.device))
            for c in range(0, len(b), eng.logit_chunk):
                bb, tt = b[c:c + eng.logit_chunk], t[c:c + eng.logit_chunk]
                lp = torch.log_softmax(eng.head_logits(xs[j][bb, tt - 1]), -1)
                lq = torch.log_softmax(eng.head_logits(xr[j][bb, tt - 1]), -1)
                kl = (lp.exp() * (lp - lq)).sum(-1)
                tok_ids = m["ids"][bb, tt][:, None]
                lr_sum += float((lp.gather(-1, tok_ids) - lq.gather(-1, tok_ids)).sum())
                kl_sum += float(kl.sum())
                n_tok += len(bb)
                seq_kl.index_add_(0, (s + bb).cpu(), kl.float().cpu())
        per_seq += seq_kl.tolist()
    res = {"adapter": a.adapter, "eval_jsonl": a.eval_jsonl, "n_answers": len(rows), "n_tokens": n_tok,
           "kl_per_token": kl_sum / max(n_tok, 1), "kl_per_answer": sum(per_seq) / max(len(per_seq), 1),
           "logratio_written_tokens": lr_sum / max(n_tok, 1), "seconds": time.perf_counter() - t0}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
