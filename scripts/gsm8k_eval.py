"""H10: did Countdown training damage other skills? GSM8K (grade-school math
word problems), the first N questions of the official test split, greedy.

Primary score (pre-registered): `correct` = the model's final answer equals
the gold answer, where the final answer is the number in the last <answer>
tag if there is a tag, else the last number in the text. So a model is judged
by its own final answer whether or not it uses the tag format; a change of
format alone does not count as better or worse. Also reported: `tag_correct`
and `last_correct` (the two parts separately).

  python scripts/gsm8k_eval.py --adapter runs/main/final --out results/eval/gsm8k_main
Writes <out>.jsonl (fields id / reward / ..., readable by compare.py) and <out>.summary.json.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.reward import ANSWER_RE  # noqa: E402
from soupgrpo.runtime import build_engine, stop_ids, tokenizer_for  # noqa: E402

INSTRUCTION = ("Solve the problem step by step. At the end, write the final answer as a single number "
               "inside <answer></answer> tags.")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load_gsm8k(n: int) -> list[dict]:
    """First n rows of the GSM8K test split. Uses the copy in data/ (hashed in
    MANIFEST.json, so the questions are fixed); downloads only if it is missing."""
    local = ROOT / "data" / "gsm8k_test_100.jsonl"
    if local.exists() and n <= 100:
        return [json.loads(line) for line in open(local)][:n]
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    out = []
    for i, row in df.head(n).iterrows():
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k-test-{i:04d}", "question": row["question"], "gold": float(gold)})
    return out


def to_num(s: str | None) -> float | None:
    if not s:
        return None
    m = NUM_RE.findall(s.replace("$", ""))
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def grade(text: str, gold: float) -> dict:
    last = to_num(text)
    tags = ANSWER_RE.findall(text)
    tag_val = to_num(tags[-1]) if tags else None
    ok = lambda v: v is not None and abs(v - gold) < 1e-6   # noqa: E731
    final = tag_val if tags else last          # the model's own final answer, whatever its format
    return {"reward": float(ok(final)), "correct": ok(final), "tag_correct": ok(tag_val), "last_correct": ok(last),
            "tag_present": bool(tags), "last_number": last, "tag_number": tag_val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    eng, mdir = build_engine(a.model, a.device, a.dtype, adapter=a.adapter)
    tok = tokenizer_for(mdir)
    stops = stop_ids(tok)
    qs = load_gsm8k(a.n)
    prompts = [tok(tok.apply_chat_template([{"role": "user", "content": f"{q['question']}\n\n{INSTRUCTION}"}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False),
                   add_special_tokens=False).input_ids for q in qs]
    t0 = time.perf_counter()
    outs = eng.generate(prompts, a.max_new, 0.0, stops, tok.pad_token_id, a.batch,
                        torch.Generator(device=a.device).manual_seed(0))
    secs = time.perf_counter() - t0
    rows = []
    for q, o in zip(qs, outs):
        text = tok.decode([t for t in o.tokens if t not in stops], skip_special_tokens=False)
        rows.append({"sample": 0, "id": q["id"], "gold": q["gold"], "completion": text, "finish": o.finish,
                     "n_tokens": len(o.tokens), **grade(text, q["gold"])})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = {"model": a.model, "adapter": a.adapter, "n": len(rows),
            "accuracy": float(np.mean([r["correct"] for r in rows])),
            "tag_accuracy": float(np.mean([r["tag_correct"] for r in rows])),
            "tag_rate": float(np.mean([r["tag_present"] for r in rows])),
            "truncated_share": float(np.mean([r["finish"] == "length" for r in rows])),
            "len_mean": float(np.mean([r["n_tokens"] for r in rows])), "seconds": secs,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
