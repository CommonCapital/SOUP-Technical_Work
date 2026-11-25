"""Re-score saved answers with the current checker (no GPU needed).

  python scripts/rescore.py results/kaggle/pilot1/pilot/*.jsonl

Prints old vs new accuracy per file, every answer whose reward changed, and
writes <file>.rescored.summary.json next to each input with the same fields
pilot_summary.py reads. Used after the E11 checker fix.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.reward import countdown_reward  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [c - h, c + h]


def main():
    for f in map(Path, sys.argv[1:]):
        ans = [json.loads(line) for line in open(f)]
        old_sum = json.loads(f.with_suffix(".summary.json").read_text())
        flips, per_prompt, new_rows = [], defaultdict(list), []
        rates = defaultdict(int)
        for a in ans:
            r = countdown_reward(a["completion"], a["numbers"], a["target"])
            if r.reward != a["reward"]:
                flips.append((a, r))
            new_rows.append({**a, **r.as_dict()})
            per_prompt[a["id"]].append(r.reward)
            for k in ("tag_present", "parsed", "numbers_correct", "value_correct"):
                rates[k] += bool(getattr(r, k))
            rates["reward"] += r.reward
        n = len(ans)
        hist = defaultdict(int)
        for v in per_prompt.values():
            hist[int(sum(v))] += 1
        P = len(per_prompt)
        s = dict(old_sum)
        s.update({
            "accuracy": rates["reward"] / n, "wilson95": wilson(int(rates["reward"]), n),
            "mixed_prompt_share": sum(c for k, c in hist.items() if 0 < k < 4) / P,
            "all_wrong_share": hist.get(0, 0) / P, "all_right_share": hist.get(4, 0) / P,
            "per_prompt_hist": dict(sorted(hist.items())),
            "tag_rate": rates["tag_present"] / n, "parsed_rate": rates["parsed"] / n,
            "numbers_rate": rates["numbers_correct"] / n, "value_rate": rates["value_correct"] / n,
            "rescored_from": str(f.with_suffix(".rescored.jsonl")), "original": str(f), "old_accuracy": old_sum["accuracy"], "n_flips": len(flips),
        })
        with open(f.with_suffix(".rescored.jsonl"), "w") as fh:
            for row in new_rows:
                fh.write(json.dumps(row) + "\n")
        out = f.with_suffix(".rescored.summary.json")
        out.write_text(json.dumps(s, indent=1))
        up = sum(1 for _, r in flips if r.reward > 0)
        print(f"{f.name}: accuracy {old_sum['accuracy']:.1%} -> {s['accuracy']:.1%} "
              f"(flips {len(flips)}: {up} up, {len(flips) - up} down), mixed {s['mixed_prompt_share']:.1%}")
        for a, r in flips[:3]:
            print(f"   {a['numbers']} -> {a['target']}: old {a['reward']} [{a['error']}] new {r.reward} expr {r.expr!r}")


if __name__ == "__main__":
    main()
