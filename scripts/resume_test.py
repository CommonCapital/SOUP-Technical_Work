"""H19: kill -9 at random moments, resume, and compare the sample-ID log with
an uninterrupted run. Pass = identical step-by-step IDs and RNG draws, zero
duplicates, zero gaps, and at least one kill landed mid-epoch.

Writes results/logs/resume_test_<timestamp>.json.
"""

from __future__ import annotations

import json
import random
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from soupgrpo.data_stream import read_id_log  # noqa: E402

PY = sys.executable
FAKE = str(ROOT / "scripts" / "fake_train.py")


def run(data, run_dir, steps, extra=()):
    return subprocess.Popen([PY, FAKE, "--data", data, "--run-dir", run_dir, "--steps", str(steps), *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def main() -> None:
    data = str(ROOT / "data" / "train_n4.jsonl")
    steps, n_kills = 200, int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed_kill = random.Random(7)
    tmp = Path(tempfile.mkdtemp(prefix="resume_test_"))

    t0 = time.time()
    ref = run(data, str(tmp / "ref"), steps)
    ref.wait()
    ref_time = time.time() - t0
    ref_log = read_id_log(tmp / "ref" / "ids.jsonl")

    # Kill targets spread over the whole run, always including the step whose
    # batch crosses the epoch boundary (1500 / 16 = 93.75 -> step 93) and one
    # kill during start-up (before the first commit).
    targets = sorted({0, 93, 94, 187, *seed_kill.sample(range(1, steps - 1), max(0, n_kills - 4))})
    state_file = tmp / "killed" / "state.json"

    def committed():
        try:
            return json.loads(state_file.read_text())["stream"]["step"]
        except (FileNotFoundError, json.JSONDecodeError):
            return 0

    kills, outputs = [], []
    p = run(data, str(tmp / "killed"), steps)
    for target in targets:
        while committed() < target and p.poll() is None:
            time.sleep(0.002)
        time.sleep(seed_kill.uniform(0, 0.03))   # random point inside the step
        if p.poll() is not None:
            break
        delay = target
        p.send_signal(signal.SIGKILL)
        p.wait()
        outputs.append(p.stdout.read())
        st = json.loads((tmp / "killed" / "state.json").read_text())["stream"] if (tmp / "killed" / "state.json").exists() else None
        kills.append({"target_step": delay, "committed_step": st and st["step"],
                      "epoch": st and st["epoch"], "position": st and st["position"]})
        p = run(data, str(tmp / "killed"), steps)
    p.wait()
    outputs.append(p.stdout.read())
    got_log = read_id_log(tmp / "killed" / "ids.jsonl")

    ref_lines = [json.loads(l) for l in (tmp / "ref" / "ids.jsonl").read_text().splitlines()]
    got_lines = [json.loads(l) for l in (tmp / "killed" / "ids.jsonl").read_text().splitlines()]
    ref_ids = [i for _, ids in ref_log for i in ids]
    got_ids = [i for _, ids in got_log for i in ids]
    epochs_seen = Counter(e for l in ref_lines for e in l["epochs"])
    # IDs repeat across epochs by design; within one epoch they must be unique.
    def per_epoch_dups(lines):
        seen, d = set(), 0
        for l in lines:
            for i in l["ids"]:
                key = (l["epochs"][0] if len(l["epochs"]) == 1 else None, i)
                d += key in seen and key[0] is not None
                seen.add(key)
        return d

    result = {
        "hypothesis": "H19",
        "steps": steps, "batch": 16, "n_records": 1500,
        "epochs_crossed": sorted(epochs_seen),
        "n_kills": len(kills), "kills": kills,
        "mid_epoch_kills": sum(1 for k in kills if k["position"] not in (None, 0)),
        "epochs_with_kills": sorted({k["epoch"] for k in kills if k["epoch"] is not None}),
        "identical_step_ids": [l["ids"] for l in ref_lines] == [l["ids"] for l in got_lines],
        "identical_rng_draws": [l["rng"] for l in ref_lines] == [l["rng"] for l in got_lines],
        "same_length": len(ref_ids) == len(got_ids),
        "steps_logged": [len(ref_lines), len(got_lines)],
        "duplicate_steps": len(got_lines) - len({l["step"] for l in got_lines}),
        "missing_steps": sorted(set(range(steps)) - {l["step"] for l in got_lines}),
        "within_epoch_duplicate_ids": per_epoch_dups(got_lines),
        "resume_messages": [o.strip().splitlines()[0] for o in outputs if o.strip().startswith("resumed")],
        "tmp_dir": str(tmp),
    }
    result["PASS"] = (result["identical_step_ids"] and result["identical_rng_draws"] and result["same_length"]
                      and result["duplicate_steps"] == 0 and not result["missing_steps"]
                      and result["within_epoch_duplicate_ids"] == 0 and result["mid_epoch_kills"] > 0)
    out = ROOT / "results" / "logs" / f"resume_test_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("kills", "resume_messages")}, indent=2))
    print("log:", out)
    sys.exit(0 if result["PASS"] else 1)


if __name__ == "__main__":
    main()
