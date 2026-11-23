"""Paired comparison of two evaluation outputs on the same questions.

  uv run python scripts/compare.py results/eval/base.jsonl results/eval/adapter.jsonl

Reports: accuracy of each, paired difference with a 95% bootstrap CI
(10,000 resamples over questions), exact McNemar p-value, and the change in
each reward sub-check (skill vs format).
Decision rule (H8): confirmed if the CI of (B - A) excludes 0.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return {r["id"]: r for r in rows if r.get("sample", 0) == 0}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(a_rows, b_rows, key="reward", n_boot=10_000, seed=0):
    ids = sorted(set(a_rows) & set(b_rows))
    x = np.array([float(a_rows[i][key]) for i in ids])
    y = np.array([float(b_rows[i][key]) for i in ids])
    d = y - x
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    return {"n": len(ids), "a": float(x.mean()), "b": float(y.mean()), "diff": float(d.mean()),
            "ci95": [float(lo), float(hi)], "a_only_right": b, "b_only_right": c,
            "mcnemar_p": mcnemar_exact(b, c), "ci_excludes_0": bool(lo > 0 or hi < 0)}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    res = {"A": sys.argv[1], "B": sys.argv[2], "reward": paired(a, b)}
    for k in ("tag_present", "parsed", "numbers_correct", "value_correct"):
        res[k] = paired(a, b, k)
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
