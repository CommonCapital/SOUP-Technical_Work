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
STAGE = __import__("os").environ.get("SOUP_STAGE", "pilot1")   # pilot1 | pilot2 | dryrun | train_main | train_control | dflash
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
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", ":!kaggle/*.ipynb"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
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

MD_INTRO_TRAIN = """# Soup GRPO task: step 3, {title} (Kaggle T4)

**Before running:** Settings, then **Accelerator: GPU T4 x2** and **Internet: On**. Then **Save Version ->
Save & Run All (Commit)** and close the tab. Takes about **{hours}**. It can run at the same time as the other
training notebook. When the version shows Complete, open its **Output** tab and download **`soup_results.zip`**.

Configuration and the SHIP rule are pre-registered in `predictions/PREDICTIONS.md` (training setup, the
amendment after dry run 1, and "Evaluation protocol and the SHIP rule"). Steps 1-5 are the usual safety checks.
{steps}
If a session ever dies mid-run: add this notebook's failed version as an **Input** (Add Input -> Your Work) and
run again; the training cell finds the saved run directory and resumes exactly (H19).
"""

MD_INTRO_DFLASH = """# Soup GRPO task: step 4, speed A/B tests + DFlash speculative decoding (Kaggle T4)

**Before running:**
1. Settings: **Accelerator: GPU T4 x2** and **Internet: On**.
2. **Add Input -> Your Work -> Notebooks:** add **soup_train_main** and **soup_train_control** (their
   finished versions). This gives this run the two trained adapters. Without them it still runs, on the base model only.
3. **Save Version -> Save & Run All (Commit)**, close the tab. Takes about **3 hours**. When it shows Complete,
   open its **Output** tab and download **`soup_results.zip`**.

Pre-registered in `predictions/PREDICTIONS.md`, "Speed A/B and DFlash". Steps 1-5 are the usual safety checks.
6. **Speed A/B (H2, H3, H5):** prefetch on/off, resident layers at equal memory, per-phase placement.
7. Download the DFlash drafter (z-lab/Qwen3-4B-DFlash-b16, 1.07 GB, weights only) and find the adapters.
8. **Greedy identity + acceptance (H21, H7, D0):** 96 held-out puzzles, main adapter (plain run too), base, control.
9. **Batch sweep (H23):** plain vs DFlash at batch 8 / 16 / 32 / 48, T = 1.
10. **Sampling test (H22):** 2,000 samples, plain vs DFlash, chi-square; biased rule as power check.
11. **Real GRPO step with and without DFlash (H6)**, A-B-B-A.
"""

DFLASH_CELLS = ['''
# 6. Speed A/B: H3 prefetch on/off, H2 resident layers at equal memory, H5 per-phase placement
q = ["--quick", "--dtype", "fp32"] if LOCAL_TEST else []
run([PY, "scripts/gpu/speed_ab.py", "--model", MODEL, *q], log="speed_ab.txt")
''', '''
# 7. DFlash drafter (weights + config only; none of its code is run) and the trained adapters
import shutil, zipfile
if LOCAL_TEST:
    from soupgrpo.dflash import write_random_drafter
    from soupgrpo.runtime import resolve_model_dir
    DRAFTER = write_random_drafter(WORK / "drafter", resolve_model_dir(MODEL), 2, (1, 2), 16, 7)
else:
    from huggingface_hub import hf_hub_download
    for f in ("config.json", "model.safetensors"):
        DRAFTER = Path(hf_hub_download("z-lab/Qwen3-4B-DFlash-b16", f)).parent
print("drafter at", DRAFTER, [p.name for p in DRAFTER.iterdir()])
ADAPTERS = {}
inp = Path("/kaggle/input")
for name in ("main", "control"):
    hits = sorted(inp.glob(f"**/adapters/{name}/adapter_config.json")) if inp.exists() else []
    if hits:
        ADAPTERS[name] = hits[-1].parent
        continue
    for z in (sorted(inp.glob("**/soup_results.zip")) if inp.exists() else []):
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if m.startswith(f"adapters/{name}/")]
            if members and f"adapters/{name}/adapter_config.json" in members:
                zf.extractall(WORK / "inputs" / z.parent.name, members)
                ADAPTERS[name] = WORK / "inputs" / z.parent.name / "adapters" / name
                break
print("adapters found:", {k: str(v) for k, v in ADAPTERS.items()} or "none (base model only)")
(REPO / "results/logs/dflash_inputs.json").write_text(json.dumps({"drafter": str(DRAFTER),
    "adapters": {k: str(v) for k, v in ADAPTERS.items()}}, indent=1))
''', '''
# 8. H21 greedy identity (main adapter, plain run in this session too), D0 / H7 acceptance (base, control)
DB = [PY, "scripts/gpu/dflash_bench.py"]
SM = ["--model", MODEL, "--drafter", DRAFTER, *(["--dtype", "fp32"] if LOCAL_TEST else [])]
N = ["--n", "4", "--batch", "2", "--max-new", "24"] if LOCAL_TEST else ["--n", "96", "--batch", "32", "--max-new", "512"]
order = [("main", True), ("base", False), ("control", False)]
for tag, plain in order:
    if tag != "base" and tag not in ADAPTERS:
        print("skip", tag, "(adapter not attached)")
        continue
    ad = ["--adapter", ADAPTERS[tag]] if tag != "base" else []
    run([*DB, "greedy", *SM, *ad, *N, *(["--plain"] if plain else []), "--ref", f"data/ref_greedy/{tag}.jsonl",
         "--tag", tag], log=f"dflash_greedy_{tag}.txt", check=False)
if "main" not in ADAPTERS:      # H21 then runs on the base model (no LoRA)
    run([*DB, "greedy", *SM, *N, "--plain", "--tag", "base_plain"], log="dflash_greedy_base_plain.txt", check=False)
''', '''
# 9. H23 batch sweep: plain vs DFlash tokens/s at T = 1, 512 new tokens (base model)
B = ["--batches", "2", "4", "--max-new", "16"] if LOCAL_TEST else ["--batches", "8", "16", "32", "48", "--max-new", "512"]
run([*DB, "sweep", *SM, *B, "--tag", "base"], log="dflash_sweep.txt", check=False)
''', '''
# 10. H22 sampling test: 2,000 samples x 16 tokens at T = 1, plain vs DFlash (current policy = main adapter)
ad = ["--adapter", ADAPTERS["main"]] if "main" in ADAPTERS else []
S = ["--samples", "60", "--batch", "30", "--dbatch", "30"] if LOCAL_TEST else ["--samples", "2000", "--batch", "48", "--dbatch", "32"]
run([*DB, "sampling", *SM, *ad, *S, "--tag", "main" if ad else "base"], log="dflash_sampling.txt", check=False)
''', '''
# 11. H6: real GRPO step 0 (same seed, same prompts) plain (A) vs DFlash (B), A-B-B-A, one step per run
if LOCAL_TEST:
    cfg = ["--prompts-per-step", "2", "--group", "4", "--max-new", "24", "--train-seqs-per-pass", "8",
           "--micro-batch", "4", "--dtype", "fp32"]
    gb = {"A": "8", "B": "4"}
else:
    cfg = ["--prompts-per-step", "6", "--group", "8", "--max-new", "512", "--train-seqs-per-pass", "24",
           "--micro-batch", "8", "--dtype", "fp16"]
    gb = {"A": "48", "B": "24"}
H6 = {"A": [], "B": []}
for i, arm in enumerate("ABBA"):
    rd = WORK / f"runs/h6_{i}_{arm}"
    extra = ["--dflash", DRAFTER] if arm == "B" else []
    rc = run([PY, "scripts/train_grpo.py", "--model", MODEL, "--data", "data/train_n5_main.jsonl", "--prompt", "v2",
              "--steps", "1", "--lr", "2e-5", "--warmup", "2", "--beta", "0", "--log-kl-every", "100", "--seed", "1234",
              "--temperature", "1.0", "--gen-batch", gb[arm], *cfg, *extra, "--run-dir", rd],
             log=f"h6_{i}_{arm}.txt", check=False)
    if rc == 0:
        m = json.loads(open(rd / "metrics.jsonl").readline())
        H6[arm].append({"time_s": m["time_s"], "reward": m["reward"], "len_mean": m["len_mean"],
                        "truncated_share": m["truncated_share"], "dflash": m.get("dflash"),
                        "smi_mib_after_gen": m.get("smi_mib_after_gen"), "gpu_gen": m.get("gpu_gen")})
    shutil.rmtree(rd, ignore_errors=True)
(REPO / "results/logs/h6_steps.json").write_text(json.dumps(H6, indent=1))
import numpy as np
for arm in "AB":
    if H6[arm]:
        print(arm, "step", np.mean([x["time_s"]["total"] for x in H6[arm]]), "s, generate",
              np.mean([x["time_s"]["generate"] for x in H6[arm]]), "s")
''']

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


# ---- training stage cells (@NAME@ = main | control)
TRAIN_CELL = '''
# 6. GRPO training, run "@NAME@" (configuration pre-registered; control differs only by --shuffle-rewards)
import shutil
NAME = "@NAME@"
RUN = REPO / "runs" / NAME
# resume support: a previous (failed) version of this notebook attached as Input
prev = sorted(Path("/kaggle/input").glob(f"**/runs/{NAME}/state.json")) if Path("/kaggle/input").exists() else []
if prev and not (RUN / "state.json").exists():
    shutil.copytree(prev[-1].parent, RUN)
    print("RESUMING from", prev[-1].parent)
if LOCAL_TEST:
    STEPS = 2
    cfg = ["--prompts-per-step", "2", "--group", "4", "--max-new", "32", "--gen-batch", "8",
           "--train-seqs-per-pass", "8", "--micro-batch", "4", "--dtype", "fp32"]
else:
    STEPS = 40
    cfg = ["--prompts-per-step", "6", "--group", "8", "--max-new", "512", "--gen-batch", "48",
           "--train-seqs-per-pass", "24", "--micro-batch", "8", "--dtype", "fp16"]
run([PY, "scripts/train_grpo.py", "--model", MODEL, "--data", "data/train_n5_main.jsonl", "--prompt", "v2",
     "--steps", STEPS, "--lr", "2e-5", "--warmup", "2", "--beta", "0", "--log-kl-every", "5", "--seed", "1234",
     "--temperature", "1.0", *cfg, *[@EXTRA@], "--run-dir", RUN], log=f"train_{NAME}.txt")
M = [json.loads(l) for l in open(RUN / "metrics.jsonl")]
assert len(M) == STEPS and [m["step"] for m in M] == list(range(STEPS)), "steps missing or duplicated"
ADAPTER = RUN / "ckpt" / f"step_{STEPS:05d}"
assert (ADAPTER / "adapter_model.safetensors").exists(), ADAPTER
shutil.copytree(ADAPTER, REPO / f"results/adapters/{NAME}", ignore=shutil.ignore_patterns("optimizer.pt", "gen_state.pt"),
                dirs_exist_ok=True)
shutil.copytree(RUN, REPO / f"results/runs/{NAME}", ignore=shutil.ignore_patterns("ckpt", ".lock"), dirs_exist_ok=True)
import numpy as np
for lo in range(0, STEPS, 10):
    w = M[lo:lo + 10]
    print(f"steps {lo}-{lo + len(w) - 1}: reward {np.mean([m['reward'] for m in w]):.3f} "
          f"trunc {np.mean([m['truncated_share'] for m in w]):.2f} zero-adv {np.mean([m['zero_adv_group_share'] for m in w]):.2f} "
          f"len {np.mean([m['len_mean'] for m in w]):.0f} adapter {w[-1]['adapter_delta_norm']:.3f} "
          f"step {np.mean([m['time_s']['total'] for m in w]):.0f}s")
print("KL to base (logged steps):", [(m["step"], m["kl_mean"]) for m in M if m.get("kl_mean") is not None])
'''

EVAL_HELDOUT_CELL = '''
# 7. Held-out evaluation of the "@NAME@" adapter: 500 new 5-number puzzles, greedy, 512 tokens
run([PY, "scripts/sample_eval.py", "--model", MODEL, "--adapter", ADAPTER, "--device", DEVICE,
     "--dtype", "fp32" if LOCAL_TEST else "fp16", "--prompt", "v2", "--data", "data/heldout_n5.jsonl",
     "--samples", "1", "--temperature", "0", "--max-new", "32" if LOCAL_TEST else "512", "--batch", "48",
     *(["--limit", "6"] if LOCAL_TEST else []), "--out", REPO / "results/eval/@NAME@"], log="eval_@NAME@.txt")
'''

KL_CELL = '''
# 8. H28: exact KL(policy || base) on the first 200 held-out answers of this adapter
run([PY, "scripts/kl_eval.py", "--model", MODEL, "--adapter", ADAPTER, "--eval-jsonl", REPO / "results/eval/@NAME@.jsonl",
     "--n", "6" if LOCAL_TEST else "200", "--dtype", "fp32" if LOCAL_TEST else "fp16",
     "--out", REPO / "results/eval/kl_@NAME@.json"], log="kl_@NAME@.txt")
'''

MERGE_CELL = '''
# 9. Merge (bf16, the shippable artifact) + H27/H29 check + held-out evaluation of the merged model
MERGED = WORK / "merged_main"
run([PY, "scripts/merge.py", "--model", MODEL, "--adapter", ADAPTER, "--out", MERGED, "--out-dtype", "bf16",
     "--device", DEVICE], log="merge_main.txt")
shutil.copy(MERGED / "merge_report.json", REPO / "results/logs/merge_report_main.json")
run([PY, "scripts/merge_check.py", "--model", MODEL, "--adapter", ADAPTER, "--merged", MERGED,
     "--data", "data/heldout_n5.jsonl", "--n", "6" if LOCAL_TEST else "50", "--max-new", "32" if LOCAL_TEST else "256",
     "--device", DEVICE, "--dtype", "fp32" if LOCAL_TEST else "fp16"], log="merge_check_main.txt", check=False)
run([PY, "scripts/sample_eval.py", "--model", MERGED, "--device", DEVICE,
     "--dtype", "fp32" if LOCAL_TEST else "fp16", "--prompt", "v2", "--data", "data/heldout_n5.jsonl",
     "--samples", "1", "--temperature", "0", "--max-new", "32" if LOCAL_TEST else "512", "--batch", "48",
     *(["--limit", "6"] if LOCAL_TEST else []), "--out", REPO / "results/eval/merged_main"], log="eval_merged_main.txt")
shutil.rmtree(MERGED)      # 8 GB; the zip must stay small. The adapter + merge.py recreate it exactly.
'''

BASE_EVAL_CELL = '''
# 9. Held-out evaluation of the BASE model (reference for every comparison), same protocol
run([PY, "scripts/sample_eval.py", "--model", MODEL, "--device", DEVICE,
     "--dtype", "fp32" if LOCAL_TEST else "fp16", "--prompt", "v2", "--data", "data/heldout_n5.jsonl",
     "--samples", "1", "--temperature", "0", "--max-new", "32" if LOCAL_TEST else "512", "--batch", "48",
     *(["--limit", "6"] if LOCAL_TEST else []), "--out", REPO / "results/eval/base"], log="eval_base.txt")
'''

GSM8K_CELL = '''
# 10. H10: GSM8K, first 100 test questions, greedy, 512 tokens (@WHO@)
run([PY, "scripts/gsm8k_eval.py", "--model", MODEL, *@ADAPTER@, "--n", "4" if LOCAL_TEST else "100",
     "--max-new", "16" if LOCAL_TEST else "512", "--batch", "48", "--dtype", "fp32" if LOCAL_TEST else "fp16",
     "--out", REPO / "results/eval/gsm8k_@WHO@"], log="gsm8k_@WHO@.txt")
'''


def md(src: str):
    CELLS.append(("markdown", src.strip("\n")))


def build_cells(b64: str, sha: str, rev: str):
    if STAGE == "train_main":
        md(MD_INTRO_TRAIN.format(title="main GRPO training run + its evaluation", hours="7 hours", steps=(
            "6. **Training: 40 GRPO steps** (~4.2 h).\n7. **Held-out evaluation** of the trained adapter: 500 new "
            "5-number puzzles, greedy.\n8. **KL** of the trained policy to the base model.\n9. **Merge** into the "
            "base weights (bf16) + merge check (H27, H29) + held-out evaluation of the merged model.\n"
            "10. **GSM8K** (100 school math problems): did other skills suffer?")))
    elif STAGE == "train_control":
        md(MD_INTRO_TRAIN.format(title="control run (shuffled rewards) + base model evaluation", hours="6.5 hours",
                                 steps=(
            "6. **Control training: 40 steps with shuffled rewards** (~4.2 h): same everything, but the reward a "
            "sequence gets is another sequence's, so no real signal. Any gain here comes from the process, not "
            "from learning the task.\n7. **Held-out evaluation** of the control adapter.\n8. **KL** of the control "
            "policy to the base model.\n9. **Held-out evaluation of the base model** (the reference for all "
            "comparisons).\n10. **GSM8K** for the base model.")))
    else:
        md({"pilot1": MD_INTRO, "pilot2": MD_INTRO_PILOT2, "dryrun": MD_INTRO_DRYRUN, "dflash": MD_INTRO_DFLASH}[STAGE])
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
    elif STAGE == "dflash":
        for cell in DFLASH_CELLS:
            guarded(cell)
    elif STAGE in ("train_main", "train_control"):
        name = STAGE.split("_")[1]
        guarded(TRAIN_CELL.replace("@NAME@", name).replace(
            "@EXTRA@", '"--shuffle-rewards"' if name == "control" else ""))
        guarded(EVAL_HELDOUT_CELL.replace("@NAME@", name))
        guarded(KL_CELL.replace("@NAME@", name))
        if name == "main":
            guarded(MERGE_CELL)
            guarded(GSM8K_CELL.replace("@WHO@", "main").replace("@ADAPTER@", '["--adapter", ADAPTER]'))
        else:
            guarded(BASE_EVAL_CELL)
            guarded(GSM8K_CELL.replace("@WHO@", "base").replace("@ADAPTER@", "[]"))
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
    if STAGE in ("pilot1", "pilot2", "dryrun"):
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
