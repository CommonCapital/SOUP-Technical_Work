# Results log

One entry per experiment, written as it happens. Failures stay in, unedited.
Predictions are copied from `predictions/PREDICTIONS.md` (committed before the run).

| ID | Date | Hypothesis | Prediction | Real result | Valid? | Verdict | If bad: cause → action | Rerun | Log |
|---|---|---|---|---|---|---|---|---|---|
| E00 | 2026-11-23 | Data checks (silent-failure rows "overlap", "unsolvable") | 0 overlap; all reference solutions score 1.0 | 0 overlap across 15 files; 10,500/10,500 solutions score 1.0 | yes (Mac) | confirmed | — | — | `scripts/make_data.py` output, `data/MANIFEST.json` |
| E01 | | H24 base difficulty | 20–60% at some level (n4 first) | not run yet | | | | | |
| E02 | | H17 L2L equivalence | fp32 ≤ 1e-5; bf16 ≤ 1.5× noise floor | not run yet | | | | | |
| E03 | 2026-11-23 | H19 exact resume (stand-in trainer loop) | killed+resumed ID log == uninterrupted | PASS: 10 kill -9 (1 during start-up, 9 mid-epoch, epochs 0 and 1, one inside the batch crossing the epoch boundary); step IDs and RNG draws identical for all 200 steps; 0 duplicates, 0 gaps | yes (Mac, fake trainer) | confirmed for the data layer | — | must be repeated with the real trainer on GPU (model + optimizer state in the same commit) | `results/logs/resume_test_*.json` |

## Entry details

### E00 — dataset generation (2026-11-23, Mac)
- Setup: `scripts/make_data.py`, seeds derived from (split, level) via SHA-256.
- Levels: n3, n4, n5 (numbers 1–99, targets 1–100), n6 (targets 100–1000, used
  as the "harder" split for n5), n3_addsub (fallback, + and − only).
- Splits per level: train 1,500 / heldout 500 / pilot 100. Train excludes every
  key in any heldout or pilot split. Key = sorted numbers + target.
- Every puzzle solvable by construction; reference solution stored and checked.
- Limit: train and test share one generator, so overfitting to the generator's
  style is not detectable here (stated in the plan's limits).

### E03 — H19 resume (2026-11-23, Mac)
- Order of operations per step: take batch → append attempted IDs → work
  (rollouts + update) → save state atomically (commit point).
- On resume: lines for uncommitted steps are dropped from the ID log (each
  kill dropped exactly 1 attempted line, as expected).
- Also checked: torn last line from kill -9 is dropped (unit test).
- Limit: resume is proven for the tested kill points only, and so far with a
  stand-in loop; the GPU trainer must save model/optimizer in the same commit.

## Facts recorded along the way
- Qwen3 tokenizer has 151,669 tokens; the model's embedding has 151,936 rows
  (padded). Logits are 151,936 wide; the budget uses 151,936.
- EOS = `<|im_end|>`, PAD = `<|endoftext|>` (different IDs, so padding is never
  confused with end of answer).
- With `enable_thinking=False` the template adds an empty
  `<think>\n\n</think>\n\n` to the prompt. Expected; outputs must contain no
  `<think>`.
- Rendered prompt length ≈ 96 tokens (4 numbers).
