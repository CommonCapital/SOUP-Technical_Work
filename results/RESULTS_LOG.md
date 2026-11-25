# Results log

One entry per experiment, written as it happens. Failures stay in, unedited.
Predictions are copied from `predictions/PREDICTIONS.md` (committed before the run).

| ID | Date | Hypothesis | Prediction | Real result | Valid? | Verdict | If bad: cause → action | Rerun | Log |
|---|---|---|---|---|---|---|---|---|---|
| E00 | 2026-11-23 | Data checks (silent-failure rows "overlap", "unsolvable") | 0 overlap; all reference solutions score 1.0 | 0 overlap across 15 files; 10,500/10,500 solutions score 1.0 | yes (Mac) | confirmed | — | — | `scripts/make_data.py` output, `data/MANIFEST.json` |
| E01 | | H24 base difficulty | 20–60% at some level (n4 first) | not run yet | | | | | |
| E02 | 2026-11-24 | H17 / H18 / H4 L2L equivalence (tiny random Qwen3, 4 layers, CPU) | fp32 ≤ 1e-5; bf16 ≤ 1.5× noise floor; greedy identical; one copy per layer per pass | fp32 logits 1.1e-6, all 56 LoRA grads ≤ 2.1e-6; bf16 logits bit-identical to HF sdpa (noise eager↔sdpa 1.6e-2), grads 5.1e-4 vs noise 5.2e-2; greedy 50/50 identical vs HF generate; copies 7 per train pass (2L−1) for B=2 and B=6 | yes (CPU only; CUDA prefetch path not yet tested) | confirmed on CPU | first run flagged 2 issues: (a) a scored token right after padding read a pad position's hidden state → fixed in `_scored_index`; (b) copy test expected 2L but the layer still in the ring buffer at the forward→backward turn is reused → test corrected to "≤ 1 per direction, independent of batch" | PASS (rerun after the copy-test fix for the case "all layers fit in the ring": PASS) | `results/logs/equivalence_20261124-000054.json`, `equivalence_20261124-021328.json` |
| E02b | | Prefetch race on CUDA | CUDA engine (prefetch, 2/3 buffers) == CPU engine fp32 | not run yet (needs GPU) | | | | | `scripts/gpu/cuda_equivalence.py` |
| E03b | 2026-11-24 | H19 with the real trainer (Qwen3-0.6B, CPU fp32) | killed + resumed run == uninterrupted run | PASS: kill -9 during step 2 (the step with an optimizer update); after resume, sample IDs, rewards, adapter norms and saved completions identical for all 4 steps | yes | confirmed (CPU) | first attempt invalid, see E06 | PASS | `results/logs/trainer_resume_20261124-020743.json` |
| E06 | 2026-11-24 | Test-harness failure (not a hypothesis) | — | first trainer kill test hung: `kill -9` hit the `uv run` wrapper, the real trainer kept running and a second (resumed) trainer wrote to the same run directory | invalid run | — | cause: `$!` was uv's PID. Fix: call `.venv/bin/python` directly and assert the PID is gone; trainer now takes an exclusive lock on its run directory, so two live trainers can never share one run | rerun = E03b | `scripts/trainer_resume_test.sh`, `scripts/train_grpo.py` |
| E07 | 2026-11-24 | H27 merged = adapter (0.6B, adapter from E05, 1 update) | greedy agreement ≥ 99%, Δlogp ≤ 3× noise | **FAIL in bf16**: greedy agreement 0/8, max Δlogp 1.95 (noise 3e-5) | yes | rejected for the bf16 artifact | diagnosis: fp32 in-memory merge == adapter (max logit diff 2e-4), so the math, names and transpose are right; the bf16 file behaves like the **base** model (same top token as base). Cause: 88% of the nonzero ΔW elements are below half a bf16 rounding step of W and vanish; relative error of the stored change 89%. Action: merge.py now reports `delta_lost_to_rounding_share` and supports `--out-dtype fp32`; on GPU measure the share for the 4B adapter (H29) | fp32 merge (`--out-dtype fp32`): lost share 0.01%, H27 PASS (greedy 8/8 identical, max Δlogp 6.3e-5 vs noise 3.0e-5), which also shows the check itself works | `results/logs/merge_check_20261124-020849.json` (bf16, FAIL), `merge_check_20261124-021146.json` (fp32, PASS) |
| E08 | 2026-11-24 | H30 dry run: fp16 mode (fp32 residual) on CPU | fp16 error ≤ plain bf16 error vs fp32 | tiny model: logits 0.26% vs bf16 2.8%, grads 0.79% vs 6.9% (nmax); Qwen3-0.6B, 3 prompts × 32 tokens: top-1 agreement 100%, mean \|Δ log p\| 0.0009, 0 non-finite | yes (CPU; the real test is 4B on the T4) | confirmed on CPU | — | — | `results/logs/fp16_check_*.json` |
| E09 | 2026-11-24 | Prompt check on Qwen3-0.6B (local, smoke test only, not the 4B pilot) | prompt v1 is fine | **v1 flawed**: one answer copied the prompt's example `(1 + 2) * 3`; answers ~20 tokens with no working; frequent `* 1` to force the value (correctly rejected: extra number). v2 (no example, brief work first): 0 copies, 0 `* 1`, ~89 tokens, 3/16 vs 1/16 correct (too few to compare), tag rate 81% vs 100% | yes (tiny sample) | v1 rejected as default | v2 written; both go to the 4B pilot as an A/B (H31, rule pre-registered) | — | `runs/mac/prompt_v1.*`, `prompt_v2.*` (not in git) |
| E04 | 2026-11-24 | Pipeline smoke test, real Qwen3-0.6B on Mac CPU (fp32) | runs end to end | pilot n3, 8 prompts × 2: 1/16 correct; tag 100%, answers ~20 tokens, no thinking; checker caught `(2 + 62 - 47) * 1` (extra number 1) | yes (smoke only, not a measurement) | works | — | — | `runs/mac/pilot_n3_06b.*` (not in git) |
| E05 | 2026-11-24 | GRPO 4 steps on 0.6B (4 prompts × G=4, lr 1e-4, 64 tokens) | adapter changes only on steps with signal | steps 0–1 all groups zero-advantage → no update (norm stays 0, as designed); step 2 one group mixed → update; step 3 behaviour shift: all 16 answers switched to step-by-step reasoning and hit the 64-token cap | yes | expected mechanics; **finding**: one Adam step moved the policy a lot | Adam's first step is ~lr × sign(grad) on every parameter, whatever the gradient size; with lr 1e-4 and one useful group it changed answer style → watch length/truncation in the GPU dry run; keep lr 2e-5 and 256-token cap | — | `runs/mac/grpo_a/` (not in git) |
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
- Kaggle host (first session, 2026-11-25): 2× Tesla T4 (15,360 MiB each; torch sees
  14,911 MiB), driver 580.178.04, CUDA 13.0, torch 2.11.0+cu130, transformers 5.20.2,
  4 CPUs, 31 GB RAM, no swap, 20 GB disk. Only GPU 0 is used.
- `torch.cuda.is_bf16_supported()` returns **True** on the T4 (capability 7.5), but
  Turing has no bf16 tensor cores; bf16 there runs emulated/slow. The fp16 decision
  (H30) does not rely on this flag.
- PCIe link shows gen 1 at idle (max gen 3 ×16): the card lowers the link speed when
  idle; `pcie_bench.py` measures the speed under load.
- Ballast check passed on the first session: 6,614 MiB held, 8,190 MiB left for us.
- First interactive run ended during the model download (cell 4 printed nothing, no
  STOPPED line): the session ended, not a failed check. Rerun as a background
  "Save & Run All" version.

## Kaggle pilot 1 (2026-11-25, notebook built from git 08afa3f)

Raw files: `results/kaggle/pilot1/` (logs, answers, summaries). Host facts are listed above.

| ID | Hypothesis | Prediction | Real result | Valid? | Verdict | If bad: cause → action | Log |
|---|---|---|---|---|---|---|---|
| E02b | Prefetch race on CUDA | CUDA engine == CPU engine (fp32) | PASS for prefetch 3 buffers, 2 buffers, 3 + copy checksums, no prefetch: logits 1.2e-6, grads 4.4e-6, greedy identical; 63/63 unit tests pass on Kaggle | yes | confirmed | — | `logs/cuda_equivalence_20261125-105700.json` |
| E10 | H30 fp16 + fp32 residual = fp32 on Qwen3-4B | top-1 ≥ 99%, mean \|Δlogp\| ≤ 0.01, 0 non-finite | top-1 99.8% (1024 positions), mean \|Δlogp\| 0.00084, max 0.020, greedy 8/8 identical, 0 non-finite; fp16 2.0× faster than fp32 | yes | **confirmed** | — | `logs/fp16_check_20261125-110308.json` |
| E12 | H13 context + workspace 0.4–0.7 GB; H15 decode pass within 20% of 7.26 GB ÷ GB/s | as stated | context + workspace **0.15 GiB** (rejected, lower: CUDA 13 loads kernels lazily); pinned 12.3 GB/s, pageable 6.95 GB/s; predicted pass 0.590 s, measured **0.603–0.612 s** over 633–1792 passes (ratio 1.02–1.04) | yes | H13 rejected (harmless), **H15 confirmed** | budget uses 0.15 GiB | `logs/pcie_20261125-105705.json`, `pilot/*.summary.json` (engine timers) |
| E12b | 8 GB cap held | our peak < 8,192 MiB | notebook printed `our_peak_mib 8291, under_8192 False`: **formula bug**, it subtracted the ballast (6,614) but not the 107 MiB that was on the card before (driver). Correct: 14,905 − 6,721 = **8,184 MiB ≤ 8,190 allowed**. Torch: peak allocated 5.8 GiB, peak reserved 7.83 GiB, i.e. ~2 GiB of the cap was allocator cache, not tensors | yes | held, but reserved memory ran up to the cap | formula fixed in the notebook; `expandable_segments:True` set from pilot 2 on; generation now splits a batch in two on OOM and logs it (`oom_splits`) | `logs/nvsmi_20261125-105539.csv`, `logs/cap_summary.json` |
| E11 | Silent-failure row "checker rejects a correct answer in another format" | pilot answers read by hand | **checker bug found.** v2 answers often say "write the final equation inside the `<answer>` tags:" and then `<answer>(62 - 47) + 2</answer>`; the pattern matched from the first `<answer>` and read `tags: <answer>(62 - 47) + 2` → rejected. Also `<answer>59 = 39 + 23 - (62 - 59)</answer>` (target on the left) was rejected. The automatic "value right but rejected" list missed both, because parsing failed before the value was computed | yes | bug, fixed | block content may not contain another `<answer>`; target accepted on either side of `=`; 3 regression tests; `pilot_summary.py` now also lists "tag present but not parsed". Rescored all 1,600 saved answers (`scripts/rescore.py`): 56 flips, **all wrong → right** | `pilot/*.rescored.summary.json` |
| E01 | H24 base difficulty 20–60% at some level; H31 v2 ≥ v1 at n4, truncation < 5% | as stated | after E11 fix: **n4/v1 13.2%** (tag 100%, ~22 tokens, no working), **n4/v2 31.2%** [26.9, 36.0], mixed 44%; n3/v2 62.3% (mixed 46%); n5/v2 6.0%. Truncation at 256 tokens: n4/v2 **65%**, n3 36%, n5 89%. Of v2 answers that finished, 79% (n4) and 82% (n3) were right | yes | H31: v2 > v1 confirmed, truncation part **rejected**. H24: n4/v2 in range, but rule 6 (truncation < 5%) fails | the cap, not arithmetic, limits v2: with a 256 cap, GRPO could "improve" just by writing less. → pilot 2 at 512 tokens (rule pre-registered as H32 before it runs). Before the fix the rule had chosen n3/v2 (53.0%); the fix moved n3 out of range | `logs/pilot_summary.txt`, `pilot/` |
