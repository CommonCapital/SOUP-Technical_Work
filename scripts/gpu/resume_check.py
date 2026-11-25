"""H19 on the GPU with the real trainer and the real model (Python port of
scripts/trainer_resume_test.sh, which needs the Mac's .venv path).

Run A: uninterrupted. Run B: kill -9 in the middle of step KILL_STEP, then
resumed. Compared per step: sample IDs (must be identical), rewards, adapter
norms and saved completions. A small configuration (few prompts, short answers)
keeps it to minutes; the code path is the training one.

On CUDA some kernels are not bit-deterministic, so after the first update the
completions of A and B may drift apart even with a correct resume. Therefore:
  PASS_exact   = IDs, rewards, norms and completions all identical
  PASS_resume  = IDs identical and step logs complete with no gaps/duplicates
                 (the data/optimizer state was restored at the right point)
Always exits 0; the verdict is in the JSON (the notebook decides what to do).

  python scripts/gpu/resume_check.py --model Qwen/Qwen3-4B --dtype fp16 --data data/train_n5_main.jsonl
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--data", default="data/train_n5_main.jsonl")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--kill-step", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "runs" / "resume_check"))
    a = ap.parse_args()

    out = Path(a.out)
    subprocess.run(["rm", "-rf", str(out)], check=True)
    out.mkdir(parents=True)
    args = [sys.executable, "scripts/train_grpo.py", "--model", a.model, "--dtype", a.dtype, "--data", a.data,
            "--steps", str(a.steps), "--prompts-per-step", "2", "--group", "4", "--max-new", "64",
            "--gen-batch", "8", "--train-seqs-per-pass", "8", "--micro-batch", "4", "--lr", "1e-4",
            "--log-kl-every", "2"]
    t0 = time.time()
    with open(out / "a.log", "w") as f:
        subprocess.run(args + ["--run-dir", str(out / "a")], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
    t_a = time.time() - t0

    with open(out / "b1.log", "w") as f:
        p = subprocess.Popen(args + ["--run-dir", str(out / "b")], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    state = out / "b" / "state.json"
    deadline = time.time() + 3600
    while time.time() < deadline:
        if p.poll() is not None:
            break
        try:
            if f'"step": {a.kill_step},' in state.read_text():
                break
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    time.sleep(3)                       # inside step KILL_STEP (after its IDs were logged)
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    killed_alive = p.poll() is None
    with open(out / "b2.log", "w") as f:
        subprocess.run(args + ["--run-dir", str(out / "b")], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, check=True)

    L = lambda q: [json.loads(line) for line in open(q)]   # noqa: E731
    ida, idb = L(out / "a/ids.jsonl"), L(out / "b/ids.jsonl")
    ma, mb = L(out / "a/metrics.jsonl"), L(out / "b/metrics.jsonl")
    sa, sb = L(out / "a/samples.jsonl"), L(out / "b/samples.jsonl")
    res = {
        "model": a.model, "dtype": a.dtype, "steps": a.steps, "kill_step": a.kill_step,
        "trainer_dead_after_kill": not killed_alive,
        "resumed_line": [line for line in open(out / "b2.log") if "resumed" in line][:1],
        "ids_identical": [x["ids"] for x in ida] == [x["ids"] for x in idb],
        "metrics_steps_b": [x["step"] for x in mb],
        "rewards": [[x["reward"] for x in ma], [x["reward"] for x in mb]],
        "adapter_norms": [[x["adapter_delta_norm"] for x in ma], [x["adapter_delta_norm"] for x in mb]],
        "samples_identical": [s["completion"] for s in sa] == [s["completion"] for s in sb],
        "seconds_run_a": t_a,
    }
    res["PASS_resume"] = (res["ids_identical"] and res["metrics_steps_b"] == list(range(a.steps))
                          and res["trainer_dead_after_kill"])
    res["PASS_exact"] = (res["PASS_resume"] and res["samples_identical"]
                         and res["rewards"][0] == res["rewards"][1] and res["adapter_norms"][0] == res["adapter_norms"][1])
    p = ROOT / "results" / "logs" / f"resume_check_{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
