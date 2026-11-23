# GPU runbook (plan steps 6–14)

Run inside `tmux` so a dropped SSH session doesn't kill anything. Every
command below writes a timestamped log under `results/` or `runs/`.

## 0. Once
```bash
git clone <repo> soup-grpo && cd soup-grpo     # or rsync from the Mac
bash scripts/gpu/setup.sh                       # env record, checks Ampere/bf16, downloads Qwen3-4B
tmux new -d -s mon 'bash scripts/gpu/monitor.sh'  # nvidia-smi + host RAM every second, whole session
```

## 1. Correctness on the GPU (before anything expensive)
```bash
uv run pytest -q                                # includes the tiny-model equivalence tests (CPU)
uv run python scripts/equivalence_test.py       # H17/H18/H4 log
```
Then the same equivalence on CUDA with prefetch on (catches prefetch races):
```bash
uv run python scripts/gpu/cuda_equivalence.py   # CUDA engine (prefetch, 2/3 buffers) == CPU engine
```

## 2. Step 6: PCIe and CUDA context (H13, H15 input)
```bash
uv run python scripts/gpu/pcie_bench.py
```

## 3. Step 7: memory and time budget (H11, H12, H1, H15, H4)
```bash
uv run python scripts/gpu/budget_check.py
```

## 4. Step 8: pilot (H24). Rule is pre-registered in predictions/PREDICTIONS.md
```bash
uv run python scripts/sample_eval.py --data data/pilot_n4.jsonl --samples 4 --temperature 1.0 --out results/pilot/n4
# then n3 or n5 depending on the rule; read 20 answers by hand if 0-5% or 90-100%
```

## 5. Step 9: dry run, 5 steps
```bash
uv run python scripts/train_grpo.py --data data/train_<LEVEL>.jsonl --run-dir runs/dry --steps 5
uv run python scripts/train_grpo.py --data data/train_<LEVEL>.jsonl --run-dir runs/dry_lr0 --steps 2 --lr 0
bash scripts/trainer_resume_test.sh Qwen/Qwen3-4B bf16 3 2          # H19 with the real model
```
Check in `metrics.jsonl`: adapter_delta_norm grows (lr>0) and stays 0 (lr=0),
zero_adv_group_share < 0.5, smi peaks < 8,192 MiB, step time vs ~6.3 min.

## 6. Step 10: main run + control
```bash
uv run python scripts/train_grpo.py --data data/train_<LEVEL>.jsonl --run-dir runs/main --steps 40
uv run python scripts/train_grpo.py --data data/train_<LEVEL>.jsonl --run-dir runs/control --steps 40 --shuffle-rewards
```
If killed, rerun the same command: it resumes.

## 7. Step 11: merge
```bash
uv run python scripts/merge.py --adapter runs/main/ckpt/step_00040 --out merged/
uv run python scripts/merge_check.py --adapter runs/main/ckpt/step_00040 --merged merged/ --data data/heldout_<LEVEL>.jsonl
```
Read `delta_lost_to_rounding_share` in `merged/merge_report.json` (H29). On the
Mac the bf16 merge lost 88% of a tiny adapter's change (E07). If H27 fails for
bf16: also merge with `--out-dtype fp32 --out merged_fp32/`, evaluate both, and
ship the adapter (or the fp32 merge), stating why.

## 8. Step 12: evaluation (same 500 held-out for every model)
```bash
for M in base adapter merged control; do ...; done   # see scripts/sample_eval.py --help
uv run python scripts/compare.py results/eval/base.jsonl results/eval/adapter.jsonl results/eval/cmp_base_adapter.json
```
