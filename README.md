# soup-grpo

GRPO + LoRA fine-tuning of Qwen3-4B on Countdown puzzles under an 8 GB GPU cap
(nvidia-smi, CUDA context included), with a self-written layer-to-layer (L2L)
streaming engine.

## Layout
- `src/soupgrpo/countdown.py`: puzzle generator (solvable by construction, difficulty knob)
- `src/soupgrpo/reward.py`: whitelist `ast` checker, exact Fractions, four sub-checks
- `src/soupgrpo/prompts.py`: Qwen3 chat template, thinking off
- `src/soupgrpo/data_stream.py`: on-disk prompt stream with exact resume
- `scripts/make_data.py`: builds and checks all splits in `data/`
- `scripts/resume_test.py`: kill -9 resume test (H19)
- `predictions/PREDICTIONS.md`: pre-registered predictions and decision rules
- `results/RESULTS_LOG.md`: one entry per experiment
- `report/`: final report draft and discussion notes

## Commands
```bash
uv sync
uv run python scripts/make_data.py
uv run pytest -q
uv run python scripts/resume_test.py 10
```
