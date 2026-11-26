# soup-grpo

GRPO + LoRA fine-tuning of Qwen3-4B on Countdown puzzles under an 8 GB GPU cap
(nvidia-smi, CUDA context included), with a self-written layer-to-layer (L2L)
streaming engine. **Verdict: DON'T SHIP** (`report/report.pdf`).

## Deliverables
| What | Where |
|---|---|
| Report (2 pages) | `report/report.pdf` (source `report/report.md`; build: `uv run --with markdown python report/build_pdf.py`) |
| Verification script + output | `uv run python scripts/final_analysis.py` → `results/final/verdict.json`, `results/final/*.json` |
| L2L equivalence test | `tests/test_l2l.py`, `src/soupgrpo/equivalence.py`, `scripts/gpu/cuda_equivalence.py`; logs `results/logs/equivalence_*.json`, `results/kaggle/*/logs/cuda_equivalence_*.json` |
| Raw logs, incl. per-second nvidia-smi and host RAM for every GPU session | `results/kaggle/{pilot1,pilot2,dryrun1,dryrun2,main,control}/logs/` |
| Pre-registered predictions and rules (git-timestamped) | `predictions/PREDICTIONS.md` |
| Experiment log (every failure kept) | `results/RESULTS_LOG.md` (E00–E17) |
| Trained adapters (PEFT format) | `results/kaggle/main/adapters/main/`, `results/kaggle/control/adapters/control/` (kept locally, not in git: 126 MB each) |
| Kaggle notebooks (self-contained, code + data embedded) | `kaggle/*.ipynb`, built by `kaggle/build_notebook.py` (`SOUP_STAGE=...`) |
| Interview notes | `report/discussion_notes.md` |

## Layout
- `src/soupgrpo/countdown.py`: puzzle generator (solvable by construction, difficulty knob)
- `src/soupgrpo/reward.py`: whitelist `ast` checker, exact Fractions, four sub-checks
- `src/soupgrpo/prompts.py`: Qwen3 chat template, thinking off
- `src/soupgrpo/data_stream.py`: on-disk prompt stream with exact resume
- `scripts/make_data.py`: builds and checks all splits in `data/`
- `scripts/resume_test.py`: kill -9 resume test (H19)
- `predictions/PREDICTIONS.md`: pre-registered predictions and decision rules
- `results/RESULTS_LOG.md`: one entry per experiment
- `src/soupgrpo/l2l.py`, `weights.py`, `lora.py`, `qwen3_math.py`: the L2L engine
- `scripts/train_grpo.py`, `sample_eval.py`, `kl_eval.py`, `gsm8k_eval.py`, `merge.py`, `merge_check.py`, `compare.py`, `independent_check.py`, `final_analysis.py`
- `report/`: final report and discussion notes

## Commands
```bash
uv sync
uv run python scripts/make_data.py
uv run pytest -q
uv run python scripts/resume_test.py 10
```
