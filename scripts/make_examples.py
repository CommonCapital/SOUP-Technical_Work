"""Readable examples from the saved held-out evaluations (no GPU, no model):
the same puzzles answered by base, main adapter and control, picked by a fixed rule
(first puzzles by id in each category), so nothing is cherry-picked by hand.

  python scripts/make_examples.py   -> results/EXAMPLES.md
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"base": "results/kaggle/control/eval/base.jsonl", "main": "results/kaggle/main/eval/main.jsonl",
        "control": "results/kaggle/control/eval/control.jsonl"}
PER_CATEGORY = 2


def load(path):
    return {r["id"]: r for r in map(json.loads, open(ROOT / path))}


def short(r):
    if r["finish"] != "eos":
        return f"cut off at {r['n_tokens']} tokens, no answer"
    expr = r.get("expr") or "(no parsable answer)"
    mark = "correct" if r["reward"] == 1.0 else "wrong"
    return f"`{expr}` — {mark} ({r['n_tokens']} tokens)"


def main():
    R = {k: load(v) for k, v in RUNS.items()}
    ids = sorted(R["base"])
    cats = [
        ("Base was cut off, main finished and got it right (the most common kind of gain)",
         lambda b, m: b["finish"] != "eos" and m["reward"] == 1.0),
        ("Both right", lambda b, m: b["reward"] == 1.0 and m["reward"] == 1.0),
        ("Base right, main wrong", lambda b, m: b["reward"] == 1.0 and m["reward"] == 0.0),
        ("Main finished but wrong", lambda b, m: m["finish"] == "eos" and m["reward"] == 0.0),
    ]
    out = ["# Examples: held-out puzzles, base vs main adapter vs control",
           "",
           "Greedy decoding, 512-token limit, prompt v2. Picked by rule (`scripts/make_examples.py`: "
           f"the first {PER_CATEGORY} puzzles by id in each category), not by hand. All 500 answers per model are in "
           "`results/kaggle/*/eval/*.jsonl`.", ""]
    for title, rule in cats:
        n_all = sum(rule(R["base"][i], R["main"][i]) for i in ids)
        out += [f"## {title} ({n_all} of 500)", ""]
        for i in [i for i in ids if rule(R["base"][i], R["main"][i])][:PER_CATEGORY]:
            b = R["base"][i]
            out += [f"**{i}**: numbers {b['numbers']}, target {b['target']}", ""]
            out += [f"- base: {short(R['base'][i])}", f"- main: {short(R['main'][i])}",
                    f"- control: {short(R['control'][i])}", ""]
            out += ["<details><summary>Main adapter's full answer</summary>", "", "```text",
                    R["main"][i]["completion"].strip(), "```", "", "</details>", ""]
    (ROOT / "results" / "EXAMPLES.md").write_text("\n".join(out))
    print("wrote results/EXAMPLES.md")


if __name__ == "__main__":
    main()
