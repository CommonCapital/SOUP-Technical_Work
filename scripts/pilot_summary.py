"""Summarise pilot runs and apply the pre-registered level-selection rule
(predictions/PREDICTIONS.md, "Pilot level-selection rule"; H24).

  python scripts/pilot_summary.py results/pilot/*.summary.json

Rule, as written before any pilot number was seen:
  - 0-5%: read 20 answers first (checker? truncation? thinking?), then go easier
  - 90-100%: check the reward is not too lax, then go harder
  - pick the level with mean in 20-60% AND the highest share of prompts with
    mixed results (1-3 of 4 correct); tie -> closer to 40%
  - truncation must be < 5% at 256 new tokens, else raise the cap
  - chosen level within 10 points of an edge (20% or 60%): add 100 prompts
Also prints, per level, answers to read by hand: rejected answers whose value
is right (possible checker false negatives), and a few random ones.
"""

import json
import random
import sys
from pathlib import Path


def band(acc):
    if acc <= 0.05:
        return "0-5%: read 20 answers, then easier"
    if acc >= 0.90:
        return "90-100%: check reward is not lax, then harder"
    if 0.20 <= acc <= 0.60:
        return "IN RANGE"
    return "outside 20-60%"


def main():
    files = [Path(f) for f in sys.argv[1:]]
    rows = []
    for f in files:
        s = json.loads(f.read_text())
        s["_file"] = str(f)
        s["_level"] = Path(s["data"]).stem.replace("pilot_", "") + "/" + s.get("prompt", "v1")
        rows.append(s)
    print(f"{'level':10s} {'acc':>6s} {'95% CI':>15s} {'mixed':>6s} {'0/4':>5s} {'4/4':>5s} {'tag':>5s} "
          f"{'value':>6s} {'trunc':>6s} {'think':>6s} {'len':>5s} {'p95':>5s} {'NaN':>4s}  verdict")
    for s in sorted(rows, key=lambda r: r["_level"]):
        lo, hi = s["wilson95"]
        print(f"{s['_level']:10s} {s['accuracy']:6.1%} {f'[{lo:.1%}, {hi:.1%}]':>15s} "
              f"{(s['mixed_prompt_share'] or 0):6.1%} {s['all_wrong_share']:5.1%} {s['all_right_share']:5.1%} "
              f"{s['tag_rate']:5.1%} {s['value_rate']:6.1%} {s['truncated_share']:6.1%} {s['thinking_share']:6.1%} "
              f"{s['len_mean']:5.0f} {s['len_p95']:5.0f} {s.get('nonfinite_events', 0):4d}  {band(s['accuracy'])}")

    ok = [s for s in rows if 0.20 <= s["accuracy"] <= 0.60]
    print()
    if ok:
        best = max(ok, key=lambda s: (s["mixed_prompt_share"] or 0, -abs(s["accuracy"] - 0.40)))
        print(f"CHOSEN by the pre-registered rule: {best['_level']} "
              f"(accuracy {best['accuracy']:.1%}, mixed {best['mixed_prompt_share']:.1%})")
        if best["truncated_share"] >= 0.05:
            print(f"  WARNING: truncation {best['truncated_share']:.1%} >= 5%: raise --max-new before training")
        if min(abs(best["accuracy"] - 0.20), abs(best["accuracy"] - 0.60)) < 0.10:
            print("  within 10 points of an edge: run 100 more prompts for this level (rule)")
    else:
        print("NO level in 20-60%. Follow the search loop: easier if all near 0, harder if near 100.")
    if any(s["thinking_share"] > 0 for s in rows):
        print("WARNING: <think> found in answers: thinking mode is not off")

    # answers to read by hand
    rng = random.Random(0)
    for s in sorted(rows, key=lambda r: r["_level"]):
        ans = [json.loads(l) for l in open(Path(s["_file"]).with_suffix("").with_suffix(".jsonl"))]
        fn = [a for a in ans if a["value_correct"] and not a["reward"]]
        print(f"\n===== {s['_level']}: value right but rejected ({len(fn)}; check for checker false negatives)")
        for a in fn[:5]:
            print(f"  {a['numbers']} -> {a['target']}: {a['expr']!r}  [{a['error']}]")
        print(f"===== {s['_level']}: 4 random answers")
        for a in rng.sample(ans, min(4, len(ans))):
            print(f"  {a['numbers']} -> {a['target']} | reward {a['reward']} | {a['finish']} | "
                  f"{a['completion'][:300]!r}")


if __name__ == "__main__":
    main()
