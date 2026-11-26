"""Final analysis and the pre-registered SHIP rule (predictions/PREDICTIONS.md,
"Evaluation protocol and the SHIP rule"). Uses only saved outputs; no GPU.

  uv run python scripts/final_analysis.py
Writes results/final/<comparison>.json and results/final/verdict.json and
prints every number the verdict uses. Running it again reproduces the verdict
from the saved files (this is the verification script).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "results" / "kaggle"
OUT = ROOT / "results" / "final"
FILES = {
    "base": K / "control/eval/base.jsonl", "main": K / "main/eval/main.jsonl",
    "control": K / "control/eval/control.jsonl", "merged": K / "main/eval/merged_main.jsonl",
    "gsm8k_base": K / "control/eval/gsm8k_base.jsonl", "gsm8k_main": K / "main/eval/gsm8k_main.jsonl",
}
PAIRS = [("main_vs_base", "base", "main"), ("control_vs_base", "base", "control"),
         ("main_vs_control", "control", "main"), ("merged_vs_base", "base", "merged"),
         ("merged_vs_main", "main", "merged"), ("gsm8k_main_vs_base", "gsm8k_base", "gsm8k_main")]


def rows(name):
    return [json.loads(line) for line in open(FILES[name])]


def compare(name, a, b):
    out = OUT / f"{name}.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/compare.py"), str(FILES[a]), str(FILES[b]), str(out)],
                   check=True, capture_output=True)
    return json.loads(out.read_text())


def fmt(d, k="reward"):
    r = d[k]
    return (f"{r['a']:.3f} -> {r['b']:.3f}  diff {r['diff']:+.3f}  CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
            f"  McNemar p {r['mcnemar_p']:.3g}  (only A right {r['a_only_right']}, only B right {r['b_only_right']})")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C = {name: compare(name, a, b) for name, a, b in PAIRS}
    for name, d in C.items():
        print(f"{name:20s} {fmt(d)}")
        for k in ("finished", "value_correct", "tag_present"):
            if k in d:
                print(f"{'':20s}   {k:14s} {fmt(d, k)}")
        if "acc_finished_A" in d:
            print(f"{'':20s}   acc|finished  A {d['acc_finished_A'][0]:.3f} (n={d['acc_finished_A'][1]})  "
                  f"B {d['acc_finished_B'][0]:.3f} (n={d['acc_finished_B'][1]})")

    kl = {n: json.loads((K / f"{n}/eval/kl_{n}.json").read_text())["kl_per_token"] for n in ("main", "control")}
    indep = {}
    for n in ("base", "main", "control", "merged"):
        p = FILES[n].with_suffix(".indep.json")
        if not p.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/independent_check.py"), str(FILES[n])],
                           check=True, capture_output=True)
        indep[n] = json.loads(p.read_text())["agree"]
    caps = {n: json.loads((K / f"{n}/logs/cap_summary.json").read_text()) for n in ("main", "control")}
    metrics = {n: [json.loads(line) for line in open(K / f"{n}/runs/{n}/metrics.jsonl")] for n in ("main", "control")}
    nonfinite = sum(json.loads((K / p).read_text()).get("nonfinite_events", 0)
                    for p in ("main/eval/main.summary.json", "main/eval/merged_main.summary.json",
                              "control/eval/control.summary.json", "control/eval/base.summary.json"))
    nonfinite += sum((m.get("nonfinite_events") or 0) for ms in metrics.values() for m in ms)

    mb, mc, cb = C["main_vs_base"], C["main_vs_control"], C["control_vs_base"]
    g = C["gsm8k_main_vs_base"]
    rule = {
        "1_H8_main_beats_base": mb["reward"]["ci95"][0] > 0,
        "2_H26_main_beats_control": mc["reward"]["ci95"][0] > 0,
        "3a_H9_acc_finished_not_lower_by_3pts": mb["acc_finished_B"][0] >= mb["acc_finished_A"][0] - 0.03,
        "3b_H9_value_correct_rises": mb["value_correct"]["diff"] > 0,
        "3c_H9_independent_checker_ge_99pct": indep["main"] >= 0.99,
        "4_H10_gsm8k_no_significant_drop": g["reward"]["ci95"][1] >= 0,
        "5_validity_cap_held": all(c["under_cap"] for c in caps.values()),
        "5_validity_no_nonfinite": nonfinite == 0,
        "5_validity_40_steps_each": all([m["step"] for m in ms] == list(range(40)) for ms in metrics.values()),
    }
    verdict = "SHIP" if all(rule.values()) else "DON'T SHIP"
    extra = {
        "H28_kl_main_gt_control": kl["main"] > kl["control"], "kl_per_token": kl,
        "control_vs_base_significant": cb["reward"]["ci95"][0] > 0,
        "independent_checker_agreement": indep, "cap_our_peak_mib": {n: c["our_peak_mib"] for n, c in caps.items()},
        "nonfinite_total": nonfinite,
        "train_reward_first10_last10": {n: [float(np.mean([m["reward"] for m in ms[:10]])),
                                            float(np.mean([m["reward"] for m in ms[-10:]]))] for n, ms in metrics.items()},
        "train_trunc_first10_last10": {n: [float(np.mean([m["truncated_share"] for m in ms[:10]])),
                                           float(np.mean([m["truncated_share"] for m in ms[-10:]]))] for n, ms in metrics.items()},
        "train_len_first10_last10": {n: [float(np.mean([m["len_mean"] for m in ms[:10]])),
                                         float(np.mean([m["len_mean"] for m in ms[-10:]]))] for n, ms in metrics.items()},
    }
    res = {"verdict": verdict, "rule": rule, "extra": extra}
    (OUT / "verdict.json").write_text(json.dumps(res, indent=2) + "\n")
    print()
    for k, v in rule.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(json.dumps(extra, indent=1))
    print(f"\nVERDICT (pre-registered rule): {verdict}")


if __name__ == "__main__":
    main()
