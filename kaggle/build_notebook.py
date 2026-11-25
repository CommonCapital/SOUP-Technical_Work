"""Build the self-contained Kaggle notebook for the baseline pilot.

The repo (code + data + predictions) is embedded as a base64 tar.gz in the
first cell, so the notebook needs no upload, no git and no SSH: import it into
Kaggle, choose GPU T4, turn Internet on, Run All, download soup_results.zip.

  uv run python kaggle/build_notebook.py
writes kaggle/soup_pilot.ipynb and kaggle/soup_pilot_cells.py (the same cells
as one script, used to test the notebook locally with SOUP_LOCAL_TEST=1).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = __import__("os").environ.get("SOUP_STAGE", "pilot1")   # pilot1 | pilot2 | dryrun
INCLUDE = ["src", "scripts", "tests", "data", "predictions", "pyproject.toml", "README.md", "GPU_RUNBOOK.md"]


def bundle() -> tuple[str, str]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in INCLUDE:
            tar.add(ROOT / name, arcname=f"soup-grpo/{name}",
                    filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


def git_rev() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", ":!kaggle/soup_pilot.ipynb"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


MD_INTRO = """# Soup GRPO task: step 1, baseline pilot on Qwen3-4B (Kaggle T4)

**Before Run All:** Settings, then **Accelerator: GPU T4 x2** (or T4) and **Internet: On**.
Takes about **2 hours**. When it finishes, download **`/kaggle/working/soup_results.zip`**
(Output panel, right side) and give it back.

What it does, in order (every step writes a timestamped log):
1. Unpacks the code and data embedded below and checks the data hashes.
2. Records the environment (GPU, driver, PCIe, RAM, torch).
3. Starts an nvidia-smi log (every second) and a **ballast process that holds everything above 8,192 MiB**,
   so this run physically cannot use more than an 8 GB card would (T4 has 16 GB).
4. Unit tests and **CUDA equivalence** of the streaming engine (prefetch on, 2 and 3 buffers).
5. **PCIe bandwidth** benchmark.
6. **fp16 check (H30):** the T4 has no bf16, so fp16 is compared with fp32 on the real Qwen3-4B.
7. **Prompt A/B (H31) and pilot (H24):** base Qwen3-4B, 100 puzzles × 4 answers. First both prompt versions at
   4 numbers, then the chosen prompt at 3 and 5 numbers (more levels only if the pre-registered rule says so),
   then the summary that picks the training level.
"""

MD_INTRO_PILOT2 = """# Soup GRPO task: step 1b, second pilot with longer answers (Kaggle T4)

**Before running:** Settings, then **Accelerator: GPU T4 x2** and **Internet: On**. Then **Save Version ->
Save & Run All (Commit)** and close the tab. Takes about **1 h 45 min**. When the version shows Complete, open
its **Output** tab and download **`soup_results.zip`**.

Why: in the first pilot, 65% of the 4-number answers were cut off at 256 tokens, so the pilot measured
"finishes in 256 tokens" more than arithmetic. The pre-registered rule says: raise the cap. This run uses
**512 new tokens** for 4 and 5 numbers (rule and predictions in `predictions/PREDICTIONS.md`, "Pilot 2"),
with the checker fixed in E11. Steps 1-5 are the same safety checks as before (8 GB cap, CUDA equivalence).
"""

MD_INTRO_DRYRUN = """# Soup GRPO task: step 2, training dry run 2 on Qwen3-4B (Kaggle T4)

**Before running:** Settings, then **Accelerator: GPU T4 x2** and **Internet: On**. Then **Save Version ->
Save & Run All (Commit)** and close the tab. Takes about **1 h 45 min**. When the version shows Complete, open
its **Output** tab and download **`soup_results.zip`**.

Everything here is pre-registered in `predictions/PREDICTIONS.md` ("Pilot 2 outcome and the training setup"):
1-5. Same safety checks as before (8 GB cap, tests, CUDA equivalence, PCIe).
6. (Memory budget at training size: done in dry run 1, E14, 8,146 MiB, under the cap; skipped here.)
7. **Resume test on the 4B** (kill -9 mid-step, resume) and an **lr = 0** check, on easy puzzles so real updates happen.
8. **Dry run: 5 real GRPO steps** with the final configuration (time per step, zero-advantage share, memory).
9. **100 extra pilot puzzles** for the chosen level (5 numbers), as the selection rule requires.
"""

CELLS: list[tuple[str, str]] = []


def code(src: str):
    CELLS.append(("code", src.strip("\n")))


def guarded(src: str):
    """Cell body runs only if no earlier step failed; any exception is recorded
    in STOP instead of stopping Run All, so the final packing cell still runs."""
    body = src.strip("\n")
    title = body.splitlines()[0]
    inner = "\n".join("        " + l if l.strip() else l for l in body.splitlines()[1:])
    CELLS.append(("code", title + "\n"
                  + "if STOP is None:\n    try:\n" + inner + "\n"
                  + "    except Exception as e:\n"
                  + "        STOP = type(e).__name__ + ': ' + str(e)\n"
                  + "        print('STOPPED:', STOP)\n"
                  + "else:\n    print('skipped because an earlier step failed:', STOP)"))


PILOT2_CELL = '''
# 7. Pilot 2 (pre-registered in predictions/PREDICTIONS.md, "Pilot 2"): prompt v2, 512 new tokens, batch 48
DTYPE = "fp32" if LOCAL_TEST else "fp16"
LIMIT = ["--limit", "4"] if LOCAL_TEST else []
MAXNEW = "48" if LOCAL_TEST else "512"
def pilot(level, prompt="v2"):
    out = REPO / f"results/pilot/{level}_{prompt}_512"
    run([PY, "scripts/sample_eval.py", "--model", MODEL, "--device", DEVICE, "--dtype", DTYPE, "--prompt", prompt,
         "--data", f"data/pilot_{level}.jsonl", "--samples", "4", "--temperature", "1.0",
         "--max-new", MAXNEW, "--batch", "48", "--out", out, *LIMIT], log=f"pilot2_{level}.txt")
    return json.loads(out.with_suffix(".summary.json").read_text())
acc = {}
for level in ["n4", "n5"]:
    s = pilot(level)
    acc[level] = s["accuracy"]
    print(f"==> {level}/v2 @512: {s['accuracy']:.1%}, mixed {s['mixed_prompt_share']:.1%}, "
          f"truncated {s['truncated_share']:.1%}, OOM splits {s.get('oom_splits')}")
(REPO / "results/pilot/choice2.json").write_text(json.dumps({"prompt": "v2", "max_new": 512, "accuracy": acc}, indent=2))
'''


DRYRUN_CELLS_ALL = ['''
# 6. Memory budget at the real training size (H11b, H12, H13): 48 answers x 512 tokens, train pass, Adam step
if not LOCAL_TEST:
    rc = run([PY, "scripts/gpu/budget_check.py", "--batch", "48", "--max-new", "512", "--micro-batch", "8",
              "--dtype", "fp16", "--data", "data/train_n5_main.jsonl", "--prompt", "v2"], log="budget.txt", check=False)
    found = sorted((REPO / "results/logs").glob("budget_*.json"))
    if rc != 0 or not found:
        # a crash of this script (not an over-cap result) must not cost the dry run: the trainer logs memory too
        print("WARNING: budget_check did not finish (exit", rc, "); continuing, see logs/budget.txt")
    else:
        B = json.loads(found[-1].read_text())
        print("our peak MiB:", B["our_peak_mib"], "| under cap:", B["under_cap"],
              "| decode pass s:", round(B["decode_pass_s"], 3))
        assert B["under_cap"], "training-size budget does not fit under the cap: do not start the dry run"
''', '''
# 7. H19 resume test on the 4B (kill -9 mid-step) + lr = 0 check
DT = "fp32" if LOCAL_TEST else "fp16"
run([PY, "scripts/gpu/resume_check.py", "--model", MODEL, "--dtype", DT], log="resume_check.txt")
R = json.loads(sorted((REPO / "results/logs").glob("resume_check_*.json"))[-1].read_text())
print("H19 PASS_resume:", R["PASS_resume"], "| PASS_exact:", R["PASS_exact"])
run([PY, "scripts/train_grpo.py", "--model", MODEL, "--dtype", DT, "--data", "data/train_n3_addsub.jsonl",
     "--prompt", "v1", "--steps", "3", "--prompts-per-step", "4", "--group", "4", "--max-new", "64",
     "--gen-batch", "16", "--train-seqs-per-pass", "16", "--micro-batch", "4", "--lr", "0", "--log-kl-every", "1",
     "--run-dir", REPO / "runs/lr0"], log="lr0.txt")
M0 = [json.loads(l) for l in open(REPO / "runs/lr0/metrics.jsonl")]
lr0 = {"adapter_delta_norm": [m["adapter_delta_norm"] for m in M0], "kl_mean": [m["kl_mean"] for m in M0],
       "grad_norm": [m["grad_norm"] for m in M0]}
# meaningful only if a gradient was actually computed (E14: all rewards 0 -> no pass -> trivially "PASS")
lr0["gradient_computed"] = any((g or 0) > 0 for g in lr0["grad_norm"])
lr0["PASS"] = (lr0["gradient_computed"] and all(v == 0 for v in lr0["adapter_delta_norm"])
               and all(abs(k) < 1e-6 for k in lr0["kl_mean"] if k is not None))
(REPO / "results/logs/lr0_check.json").write_text(json.dumps(lr0, indent=2))
print("lr=0 check:", lr0)
''', '''
# 8. Dry run: 5 GRPO steps with the final configuration (H1, H25, H33, H11b)
if LOCAL_TEST:
    cfg = ["--steps", "2", "--prompts-per-step", "2", "--group", "4", "--max-new", "32", "--gen-batch", "8",
           "--train-seqs-per-pass", "8", "--micro-batch", "4", "--dtype", "fp32"]
else:
    cfg = ["--steps", "5", "--prompts-per-step", "6", "--group", "8", "--max-new", "512", "--gen-batch", "48",
           "--train-seqs-per-pass", "24", "--micro-batch", "8", "--dtype", "fp16"]
run([PY, "scripts/train_grpo.py", "--model", MODEL, "--data", "data/train_n5_main.jsonl", "--prompt", "v2",
     "--lr", "2e-5", "--warmup", "2", "--log-kl-every", "2", "--seed", "1234", *cfg,
     "--run-dir", REPO / "runs/dryrun"], log="dryrun.txt")
M = [json.loads(l) for l in open(REPO / "runs/dryrun/metrics.jsonl")]
for m in M:
    t = m["time_s"]
    print(f"step {m['step']}: {t['total']:.0f}s (generate {t['generate'] / t['total']:.0%}) reward {m['reward']:.2f} "
          f"acc|finished {m['acc_finished']} trunc {m['truncated_share']:.0%} zero-adv {m['zero_adv_group_share']:.0%} "
          f"adapter {m['adapter_delta_norm']:.2e} kl {m['kl_mean']} smi {m['smi_mib_after_gen']}/{m['smi_mib_after_train']}")
import shutil
shutil.copytree(REPO / "runs/dryrun", REPO / "results/dryrun", ignore=shutil.ignore_patterns("ckpt"), dirs_exist_ok=True)
''', '''
# 9. Pilot rule 4: 100 extra n5 puzzles (taken out of the training data), prompt v2, 512 tokens
run([PY, "scripts/sample_eval.py", "--model", MODEL, "--device", DEVICE, "--dtype", "fp32" if LOCAL_TEST else "fp16",
     "--prompt", "v2", "--data", "data/pilot_n5_extra.jsonl", "--samples", "4", "--temperature", "1.0",
     "--max-new", "48" if LOCAL_TEST else "512", "--batch", "48", "--out", REPO / "results/pilot/n5_v2_512_extra",
     *(["--limit", "4"] if LOCAL_TEST else [])], log="pilot_n5_extra.txt")
''']
# Dry run 2: the budget cell already ran in dry run 1 (E14: 8,146 MiB, under the cap), so it is skipped.
DRYRUN_CELLS = DRYRUN_CELLS_ALL[1:]


def md(src: str):
    CELLS.append(("markdown", src.strip("\n")))


def build_cells(b64: str, sha: str, rev: str):
    md({"pilot1": MD_INTRO, "pilot2": MD_INTRO_PILOT2, "dryrun": MD_INTRO_DRYRUN}[STAGE])
    code(f'''
# 1. Unpack code + data (git {rev}, bundle sha256 {sha[:16]})
import base64, hashlib, io, json, os, subprocess, sys, tarfile, time
from pathlib import Path
LOCAL_TEST = os.environ.get("SOUP_LOCAL_TEST") == "1"   # only for testing this notebook on a laptop
WORK = Path(os.environ.get("SOUP_WORK", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)
BUNDLE = "{b64}"
raw = base64.b64decode(BUNDLE)
assert hashlib.sha256(raw).hexdigest() == "{sha}", "bundle corrupted"
with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
    tar.extractall(WORK, **({{"filter": "data"}} if hasattr(tarfile, "data_filter") else {{}}))
REPO = WORK / "soup-grpo"
os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))
man = json.loads((REPO / "data/MANIFEST.json").read_text())
for f, info in man["files"].items():
    assert hashlib.sha256((REPO / "data" / f).read_bytes()).hexdigest() == info["sha256"], f
(REPO / "results/logs").mkdir(parents=True, exist_ok=True)
(REPO / "results/pilot").mkdir(parents=True, exist_ok=True)
print("code at", REPO, "| data files verified:", len(man["files"]), "| bundle git rev {rev}")
''')
    code('''
# helper: run a command, stream its output live, keep a copy in results/logs
def run(cmd, log=None, check=True, env=None):
    print("\\n$", " ".join(map(str, cmd)), flush=True)
    t0 = time.time()
    p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})})
    lines = []
    for line in p.stdout:
        if "Warning" in line or "warn(" in line:
            continue
        print(line, end="", flush=True)
        lines.append(line)
    p.wait()
    if log:
        (REPO / "results/logs" / log).write_text("".join(lines))
    print(f"[exit {p.returncode}, {time.time() - t0:.0f}s]", flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return p.returncode

PY = sys.executable
STOP = None   # set by the first failing step; later steps are skipped, results are still packed
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # one GPU, as in the task
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # less cache fragmentation (pilot 1: 2 GiB)
MODEL = os.environ.get("SOUP_MODEL") or ("Qwen/Qwen3-0.6B" if LOCAL_TEST else "Qwen/Qwen3-4B")   # SOUP_MODEL: local tests only
DEVICE = "cpu" if LOCAL_TEST else "cuda"
''')
    code('''
# 2. Environment + dependencies
if not LOCAL_TEST:
    run([PY, "-m", "pip", "install", "-q", "transformers>=4.51", "safetensors>=0.4", "huggingface_hub"], check=False)
    TS = time.strftime("%Y%m%d-%H%M%S")
    env_txt = subprocess.run("nvidia-smi; nvidia-smi --query-gpu=name,memory.total,driver_version,"
                             "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max "
                             "--format=csv; free -g; nproc; df -h /kaggle/working",
                             shell=True, capture_output=True, text=True).stdout
    import torch, transformers
    env_txt += f"\\ntorch {torch.__version__} cuda {torch.version.cuda} transformers {transformers.__version__}"
    env_txt += f"\\ndevice {torch.cuda.get_device_name(0)} capability {torch.cuda.get_device_capability(0)}"
    env_txt += f" bf16_supported {torch.cuda.is_bf16_supported()}"
    (REPO / f"results/logs/env_{TS}.txt").write_text(env_txt)
    print(env_txt)
''')
    guarded('''
# 3. Session monitor (nvidia-smi + host RAM every second) and the 8 GB ballast
if not LOCAL_TEST:
    TS = time.strftime("%Y%m%d-%H%M%S")
    smi_log = open(REPO / f"results/logs/nvsmi_{TS}.csv", "w")
    monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,"
                                "pcie.link.gen.current,pcie.link.width.current,temperature.gpu,power.draw,clocks.sm",
                                "--format=csv", "-l", "1"], stdout=smi_log, stderr=subprocess.STDOUT)
    host_mon = subprocess.Popen(["bash", "-c", f"while true; do echo \\"$(date -Is),$(free -m | awk '/Mem:/ {{print $3\\",\\"$7}}')\\""
                                 f" >> {REPO}/results/logs/hostmem_{TS}.csv; sleep 1; done"])
    ballast = subprocess.Popen([PY, "scripts/gpu/ballast.py", "--cap-mib", "8192"],
                               stdout=open(REPO / "results/logs/ballast.out", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        if (REPO / "results/logs/ballast.json").exists():
            break
        time.sleep(1)
    BALLAST = json.loads((REPO / "results/logs/ballast.json").read_text())
    print("ballast:", BALLAST)
    assert BALLAST["free_left_mib"] <= 8192, "cap not in place"
''')
    guarded('''
# 4. Model download (~8 GB) and correctness of the engine on this GPU
t0 = time.time()
from soupgrpo.runtime import resolve_model_dir
print("downloading", MODEL, "(~8 GB, a few minutes; no output until done) ...", flush=True)
print("model at", resolve_model_dir(MODEL), f"({time.time() - t0:.0f}s)")
run([PY, "-m", "pytest", "-q", "-x"], log="pytest.txt", check=False)
if not LOCAL_TEST:
    run([PY, "scripts/gpu/cuda_equivalence.py"], log="cuda_equivalence.txt")
''')
    guarded('''
# 5. PCIe bandwidth (pinned vs pageable, one 4B layer = 202 MB) + CUDA context cost
if not LOCAL_TEST:
    run([PY, "scripts/gpu/pcie_bench.py"], log="pcie.txt")
''')
    if STAGE == "pilot1":
      guarded('''
# 6. H30: fp16 vs fp32 on the real model (T4 has no bf16). Stop here if it fails.
n, mx = (3, 32) if LOCAL_TEST else (8, 128)
run([PY, "scripts/gpu/fp16_check.py", "--model", MODEL, "--device", DEVICE, "--n", n, "--max-new", mx],
    log="fp16_check.txt")
H30 = json.loads(sorted((REPO / "results/logs").glob("fp16_check_*.json"))[-1].read_text())
print("H30 PASS:", H30["PASS"])
assert H30["PASS"], "fp16 changes the model's behaviour: do not pilot in fp16 (see results/logs)"
if os.environ.get("SOUP_FORCE_FAIL") == "fp16":   # local test of the failure path only
    raise AssertionError("forced failure (test)")
''')
    if STAGE == "pilot2":
        guarded(PILOT2_CELL)
    elif STAGE == "dryrun":
        for cell in DRYRUN_CELLS:
            guarded(cell)
    else:
      guarded('''
# 7. Pilot (H24) + prompt A/B (H31): base model, 100 puzzles x 4 answers, T=1.0, 256 new tokens, fp16
DTYPE = "fp32" if LOCAL_TEST else "fp16"
LIMIT = ["--limit", "6"] if LOCAL_TEST else []
MAXNEW = "48" if LOCAL_TEST else "256"
def pilot(level, prompt):
    out = REPO / f"results/pilot/{level}_{prompt}"
    run([PY, "scripts/sample_eval.py", "--model", MODEL, "--device", DEVICE, "--dtype", DTYPE, "--prompt", prompt,
         "--data", f"data/pilot_{level}.jsonl", "--samples", "4", "--temperature", "1.0",
         "--max-new", MAXNEW, "--batch", "64", "--out", out, *LIMIT], log=f"pilot_{level}_{prompt}.txt")
    return json.loads(out.with_suffix(".summary.json").read_text())

# H31 prompt A/B at the middle level (rule pre-registered in predictions/PREDICTIONS.md)
ab = {p: pilot("n4", p) for p in ("v1", "v2")}
def in_range(s): return 0.20 <= s["accuracy"] <= 0.60
if in_range(ab["v1"]) != in_range(ab["v2"]):
    PROMPT = "v1" if in_range(ab["v1"]) else "v2"
elif in_range(ab["v1"]):
    PROMPT = "v1" if ab["v1"]["mixed_prompt_share"] > ab["v2"]["mixed_prompt_share"] else "v2"
else:
    PROMPT = "v1" if ab["v1"]["accuracy"] > ab["v2"]["accuracy"] else "v2"
print({p: round(s["accuracy"], 3) for p, s in ab.items()}, "-> prompt", PROMPT)

acc = {"n4": ab[PROMPT]["accuracy"]}
for level in ["n3", "n5"]:              # both neighbours of the middle level (pre-registered order)
    acc[level] = pilot(level, PROMPT)["accuracy"]
    print(f"==> {level}: {acc[level]:.1%}")
# search loop from the plan: only if no level landed in 20-60%
if not any(0.20 <= a <= 0.60 for a in acc.values()):
    if all(a >= 0.90 for a in acc.values()):
        acc["n6"] = pilot("n6", PROMPT)["accuracy"]                # harder: 6 numbers, targets 100-1000
    elif all(a <= 0.20 for a in acc.values()):
        acc["n3_addsub"] = pilot("n3_addsub", PROMPT)["accuracy"]  # easier: only + and -
(REPO / "results/pilot/choice.json").write_text(json.dumps({"prompt": PROMPT, "accuracy": acc}, indent=2))
print(acc)
''')
    guarded('''
# 8. Summary with the pre-registered selection rule + answers to read by hand
run([PY, "scripts/pilot_summary.py", *sorted(map(str, (REPO / "results/pilot").glob("*.summary.json")))],
    log="pilot_summary.txt")
''')
    code('''
# 9. Peak memory of THIS run (nvidia-smi used minus the ballast) and pack results (always runs)
if not LOCAL_TEST and "smi_log" in globals() and "BALLAST" in globals():
    import csv
    rows = list(csv.reader(open(smi_log.name)))[1:]
    used = [int(r[1].strip().split()[0]) for r in rows if len(r) > 1 and r[1].strip().split()[0].isdigit()]
    # subtract everything that was on the card before our processes started (ballast + driver baseline);
    # pilot 1 subtracted only the ballast and over-reported by the 107 MiB baseline (E12)
    peak = max(used) - BALLAST["smi_used_after_mib"]
    cap = {"smi_peak_total_mib": max(used), "smi_before_us_mib": BALLAST["smi_used_after_mib"],
           "ballast_used_mib": BALLAST["ballast_used_mib"], "our_peak_mib": peak,
           "cap_mib": BALLAST["cap_mib"], "under_cap": peak <= BALLAST["cap_mib"], "samples": len(used)}
    (REPO / "results/logs/cap_summary.json").write_text(json.dumps(cap, indent=2))
    print(cap)
import shutil
(REPO / "results/logs/stop_reason.txt").write_text(str(STOP))
print("STOP reason:", STOP)
zip_base = WORK / "soup_results"
shutil.make_archive(str(zip_base), "zip", REPO / "results")
print("DOWNLOAD:", str(zip_base) + ".zip")
''')


def main():
    b64, sha = bundle()
    rev = git_rev()
    build_cells(b64, sha, rev)
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        c = {"cell_type": kind, "metadata": {}, "source": src, "id": f"c{i}"}
        if kind == "code":
            c.update({"execution_count": None, "outputs": []})
        nb["cells"].append(c)
    name = "soup_pilot" if STAGE == "pilot1" else f"soup_{STAGE}"
    out = ROOT / "kaggle" / f"{name}.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    script = "\n\n".join(src for kind, src in CELLS if kind == "code")
    (ROOT / "kaggle" / f"{name}_cells.py").write_text(script + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), bundle sha256 {sha[:16]}, git {rev}")


if __name__ == "__main__":
    main()
