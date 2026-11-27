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

## Kaggle pilot 2 (2026-11-25, notebook built from git 92ddcae, 512 new tokens)

Raw files: `results/kaggle/pilot2/`.

| ID | Hypothesis | Prediction | Real result | Valid? | Verdict | If bad: cause → action | Log |
|---|---|---|---|---|---|---|---|
| E13 | H32 + pilot-2 selection rule | n4/v2 40–55%, truncation 10–35%; n5/v2 10–25% | **n4/v2 56.2%** [51.4, 61.0], mixed 44%, truncation 38.8%, accuracy among finished 91.8%; **n5/v2 21.5%** [17.8, 25.8], mixed 47%, truncation 70.8%, accuracy among finished 73.5%. Rule → **n5/v2**. Checks: CUDA equivalence PASS again, 66/66 tests, our memory peak **7,480 MiB** (cap formula fixed; torch peak allocated 7.05 GiB, reserved 7.14 GiB: expandable segments removed the ~2 GiB cache gap), 0 OOM splits, decode pass 0.61 s | yes | H32: n5 confirmed; n4 just above (56.2 vs 55) and its truncation above (38.8 vs 35) → rejected, small misses | n5 is within 10 points of 20% → 100 extra prompts (rule 4), in the dry-run notebook. n5 is the better training set beyond the mixed share: n4's finished answers are 92% right (little arithmetic left to learn, mostly length), n5's are 74% right. Risk: 71% truncation, so H9 is judged on accuracy among finished answers too (rule 3) | `logs/pilot_summary.txt`, `pilot/*_512.summary.json`, `logs/cap_summary.json` |

## Kaggle dry run 1 (2026-11-25, notebook built from git 452d051)

Raw files: `results/kaggle/dryrun1/`.

| ID | Hypothesis | Prediction | Real result | Valid? | Verdict | If bad: cause → action | Log |
|---|---|---|---|---|---|---|---|
| E14 | H11b, H12, H14, H1, H25, H33, H19 at the training configuration | see "Pilot 2 outcome and the training setup" | **Crashed at step 1 of 5: host RAM, not GPU.** GPU (budget_check, 48×512): our peak **8,146 MiB** (under the 8,190 allowed; generation 7.17 GiB allocated, training pass 3.96 GiB). H12: unchunked logits (8,192 rows) **fit** (6.67 GiB) → rejected, harmless. **Host RAM peak 30.9 of 31 GB** in every training pass (H14 predicted 13–17 GB → **rejected**): the per-layer input store for 48 sequences is ~11.5 GB fp32, and growing it held up to three copies (old pinned block kept by PyTorch's pinned cache, pageable temporary, new pinned copy). Step 0 completed: **434 s** (generation 321 s = **74%**), reward 0.19, **zero-advantage groups 17%**, 40/48 sequences trained, truncation 75%, grad norm 0.063. The resume and lr = 0 checks "passed" but **proved little**: on n5 with 64 tokens every reward was 0, so no update ever happened | partly | H1 (70–90%) confirmed with 1 step; H25 (< 50%) confirmed with 1 step; H33 (5.5–8 min) confirmed with 1 step; H11b confirmed (GPU); H14 rejected; H19 on the 4B **not shown** (no update to restore) | fix: buffer allocated pinned directly, pinned cache released before any growth, trainer reserves the largest size once at start (`boundary_allocations` logged per step), test added; **24 sequences per training pass** (2 passes per step, same math: gradients accumulate, constant normaliser) → ~5.7 GB host; resume and lr = 0 checks now use easy puzzles (n3_addsub, prompt v1) and fail unless a real update happened | `logs/hostmem_20261125-151923.csv`, `logs/budget_20261125-152912.json`, `logs/dryrun.txt`, `logs/resume_check_20261125-153527.json` |

## Kaggle dry run 2 (2026-11-25, notebook built from git 351e92c)

Raw files: `results/kaggle/dryrun2/`.

| ID | Hypothesis | Prediction | Real result | Valid? | Verdict | If bad: cause → action | Log |
|---|---|---|---|---|---|---|---|
| E15 | H19 on the 4B; lr = 0; H1, H25, H33, H11b, H14b at the training configuration; pilot rule 4 | as pre-registered | **Resume: PASS_exact** with a real update (adapter norm 0.240 restored, kill -9 during step 1, IDs/rewards/answers identical); steps 1–2 had no update, so restored Adam moments were not exercised on the 4B (they were on 0.6B, E03b). **lr = 0: PASS** (gradients 0.32 and 0.19 computed, adapter stayed 0, KL 0.0). **5 GRPO steps, no crash:** step time 401/396/335/384/386 s (mean of 1–4: **375 s → 40 steps ≈ 4.2 h**), generation 79–95% of step time, zero-advantage groups 17/0/83/17/33% (mean 30%), reward 0.19/0.40/0.06/0.31/0.19 (6 prompts per step: noisy), accuracy among finished 0.75–0.88, adapter norm 0.05 → 0.23, KL to base 0.0009 → 0.0020, GPU peak 8,158 MiB, torch allocated ≤ 4.05 GiB, **boundary buffer allocated once** (5.14 GiB), **host RAM peak 22.05 GiB** (9.5 GiB still free). **Extra 100 n5 prompts: 25.5%** [21.5, 30.0], mixed 56% → n5 over 200 prompts **23.5%**, stays | yes | H19 confirmed (4B, with the limit stated); lr = 0 confirmed; H1 confirmed (79–95%); H25 confirmed on average (30%; one step 83%); H33 confirmed (6.25 min); H11b confirmed; H14b (≤ 22 GB) rejected by 0.05 GiB, harmless (9.5 GiB free); level n5 confirmed | — | `dryrun/metrics.jsonl`, `logs/resume_check_*.json`, `logs/lr0_check.json`, `logs/hostmem_*.csv`, `pilot/n5_v2_512_extra.summary.json` |

## Kaggle main training run (2026-11-25/26, notebook built from git e791de4)

Raw files: `results/kaggle/main/` (adapter safetensors kept locally, not in git: 126 MB).
Base-model held-out evaluation comes from the control notebook; the verdict waits for it.

| ID | Hypothesis | Prediction | Real result | Valid? | Verdict | If bad: cause → action | Log |
|---|---|---|---|---|---|---|---|
| E16 | Main run mechanics; H25, H33, H28 (partial), H27, H29, H9(c); held-out accuracy of the adapter | as pre-registered | **40/40 steps, no crash, no resume needed.** Step time mean 365 s (4.05 h). Training reward by tens of steps: 0.23 / 0.13 / 0.28 / 0.32; linear trend over 40 steps **+0.18** (bootstrap over steps 95% CI [+0.06, +0.31]); truncation flat (0.68 → 0.65), length flat (434 → 443 tokens), **accuracy among finished training answers 0.70 → 0.90**. Zero-advantage groups mean 46% (one step 100%). KL to base on training batches 0 → 0.003 (step 5) → 0.02–0.14 (steps 10–35). Adapter norm 0.05 → 0.76. **Held-out (500, greedy): adapter 24.8%** [21.2, 28.8], truncation 70.4%, accuracy among finished 83.8% (148 finished). Exact KL(adapter ‖ base) on its own held-out answers: **0.065 per token**, 29.6 per answer. **Merge: bf16 lost 87.0% of the change** (relative error 86%), merge check **FAIL** (greedy agreement 0/50, max Δlogp 2.46 vs noise 0.06), merged model 26.2% held-out (paired vs adapter: +1.4 points, CI [−2.8, +5.6], 117 of 500 puzzles differ). GSM8K with the adapter: 92% (tag 69%). Cap held: 8,158 MiB; host RSS ≤ 19.6 GB; 0 non-finite values; boundary buffer allocated once. **Independent checker (H9c): 100% agreement on all 2,600 saved answers** (main, merged, pilot 2, rescored pilot 1); on the *pre-fix* pilot-1 scores it disagrees on exactly the 52 answers E11 corrected, so it would have caught that bug by itself | yes | H25 rejected narrowly (mean 46% < 50% holds on average, but several steps ≥ 50%); H33 confirmed; **H29 rejected → the merged bf16 file is not the trained model: ship the adapter** (pre-registered fallback); H27 fails for bf16 (expected after H29); H9(c) confirmed. H8/H26/H9(a,b)/H10/H28 wait for the control notebook (base and control evaluations) | merged bf16 ≈ base, as E07 warned: ΔW per element ~1e-5–1e-4, below half a bf16 step of the weights. Ship the adapter (or an fp32 merge, 2× size) | `runs/main/metrics.jsonl`, `eval/*.summary.json`, `eval/kl_main.json`, `logs/merge_report_main.json`, `logs/merge_check_20261126-013358.json`, `eval/*.indep.json` |

## Kaggle control run + base evaluation, and the verdict (2026-11-26)

Raw files: `results/kaggle/control/`; every number below is reproduced by
`uv run python scripts/final_analysis.py` (writes `results/final/*.json`, `verdict.json`).

| ID | Hypothesis | Prediction | Real result | Valid? | Verdict | If bad: cause → action | Log |
|---|---|---|---|---|---|---|---|
| E17 | H8, H26, H9, H10, H28 + validity; the SHIP rule | pre-registered | Held-out n5, 500 puzzles, greedy, paired: **base 19.2%**, **main 24.8%** (+5.6 points, CI [+1.2, +10.0], McNemar p 0.017), **control (shuffled rewards) 28.8%** (+9.6, CI [+5.4, +13.8], p 1.4e-5), **main − control −4.0** (CI [−8.4, +0.4], p 0.10), merged bf16 26.2% (+7.0 vs base). The gains are mostly **finishing within 512 tokens**: finished share base 21.2% → main 29.6% → control 33.6%; accuracy among finished answers *fell* (base 90.6% of 106 → main 83.8% of 148 → control 85.7% of 168; the newly finished answers are harder ones). GSM8K: base 94%, main 92% (−2, CI [−6, +2]). KL(adapter ‖ base) per token on own answers: **main 0.065 > control 0.029** (H28 confirmed: real reward moved the policy more, without more accuracy). Training curves: both runs saw **identical prompts per step**; training reward rose the same in both (main +0.18 over 40 steps, control +0.22), accuracy among finished training answers rose in both (main 0.70 → 0.90, control 0.59 → 0.73). Independent checker 100% on base/main/control/merged. Validity: cap 8,158 MiB in both, 0 non-finite, 40/40 steps each | yes | **H8 confirmed** (+5.6, inside the predicted +5–15); **H26 rejected** (control improved, more than main); **H9 fails (a)** (accuracy among finished fell 6.8 points), (b) and (c) pass; **H10 confirmed**; **H28 confirmed**. **SHIP rule: DON'T SHIP** (rules 2 and 3a fail) | cause (best reading, one seed per arm): the 512-token cap binds for ~79% of base answers, so any change that makes answers finish more often raises the score; a no-signal adapter did that as well as the real one. **Correction to E16:** the rising training reward and "accuracy among finished 0.70 → 0.90" were not evidence of learning: the control, with the same prompt order, shows the same rise | `results/final/verdict.json`, `results/final/*.json` |

## E18 — DFlash on the Mac (tiny model, CPU): an intermittent NaN in speculative decoding (2026-11-26)

Own re-implementation of the DFlash drafter + batched verification (`src/soupgrpo/dflash.py`),
tests in `tests/test_dflash.py`: greedy output identical to plain L2L greedy (random, oracle and
half-oracle drafters, stop tokens, LoRA on/off); the acceptance rule reproduces p exactly on a
synthetic 6-token example (400k draws); end-to-end sampling frequencies match plain sampling
and the deliberately biased rule is rejected.

**Bug found by the local notebook test:** the `--dflash` trainer run crashed twice with
"probability tensor contains nan", while the same command run by hand passed. Cause: rows accept
different numbers of tokens, so the verify pass reads KV slots up to the *longest* row; for shorter
rows some of those slots were never written. They are masked, but SDPA adds the mask (−inf) to
q·k, and allocator garbage that happens to be NaN gives NaN + (−inf) = NaN. Whether it fires
depends on what memory `torch.empty` hands out, hence intermittent. In greedy mode this would have
been **silent** (argmax of NaN logits = token 0, no crash). Fix: KV cache zero-initialised.
Regression test fills every `torch.empty` with NaN and makes rows drift apart (fails before the
fix, passes after). The plain engine is not affected (it never reads past what it wrote).

## E19 — the OOM fallback could never recover (found in the Kaggle sweep, 2026-11-26)

H23 sweep, batch 48: the plain arm ran out of memory (the drafter was still on the GPU, so 48
rows did not fit). The fallback splits the batch in two and retries, but every retry also ran
out, down to batch 1 ("Tried to allocate 88 MiB … 7.73 GiB allocated"). Cause: the retry was
called **inside** the `except` block. While it runs, the live exception's traceback keeps the
failed attempt's frame alive, and with it that attempt's KV cache (4.1 GiB at batch 48); each
split level stacked another failed attempt on top. Same pattern in `DFlashGenerator._safe`.
Fix: leave the `except` block first, then empty the cache and split. Regression test
`test_oom_split_frees_failed_attempt[plain|dflash]` fakes an OOM after allocating a stand-in
"KV cache" and asserts it is freed before each retry (fails before the fix, passes after).
**Earlier results are unaffected:** no saved run ever reported an OOM split.

## Kaggle speed A/B + DFlash run (2026-11-26, notebook soup_dflash, git 5e85e72)

Raw files: `results/kaggle/dflash/logs/`. One T4 session, 8 GB cap held (our peak 8,178 MiB),
tests 75/75, CUDA equivalence PASS, STOP none. **The adapters were not attached** ("adapters
found: none"), so everything DFlash ran on the base model only; the main/control parts wait
for the follow-up notebook `soup_dflash2`.

| ID | Hypothesis | Prediction | Real result | Valid? | Verdict | If bad: cause → action | Log |
|---|---|---|---|---|---|---|---|
| E20 | H3 prefetch | train: hides > 80% of copy time; decode: < 20%, and decode pass with prefetch within 10% of t_copy | t_copy 16.3 ms/layer, 0.588 s per pass. **Decode** (B 48, KV 712): on 0.604 s, off 0.785 s (all 8 runs within 1 ms) → hidden share **31%**; pass with prefetch = t_copy + 2.7%. **Train** phase (2 × 24 seqs + Adam): on 85.62 s, off 85.79 s, copies 2.35 s → hidden share **7%** | yes (A-B-B-A ×2, spread ≤ 0.3 s) | decode < 20%: **rejected** (31%); decode ≈ t_copy: **confirmed**; train > 80%: **rejected** (7%) | decode: I assumed 1–2 ms compute per layer; real ≈ 5.5 ms (B 48 with 712-token attention + LoRA), and prefetch hides all of it behind the copy, so decode runs at the PCIe floor. Train: copies are only 2.7% of the phase, and turning prefetch off adds only 0.16 s, not 2.35 s. The off-arm copies sit on the compute stream, so they can only be free if that stream has ≥ 2.35 s of idle time: best reading, the training phase is host-bound (≈ 1e15 FLOP in 85.6 s ≈ 11 TFLOP/s, well under T4 fp16 matmul speed; per-chunk `.cpu()` syncs in the head loop). **Not profiled** → profile before claiming either way | `speed_ab_20261126-173227.json` |
| E20 | H2 equal memory | R 0 + B 48 beats R 8 + B 32 by 1.10–1.25× (model 1.17); greedy tokens of the 32 shared prompts identical | **79.5 vs 63.9 tok/s = 1.24×** (peak 6.90 vs 6.58 GiB); pass 0.604 vs 0.501 s (model 0.47 s). Tokens identical: **False** | throughput yes; identity check failed | **ratio confirmed**; identity sub-check **failed** | the two arms use different batch sizes (48 vs 32), so fp16 matmul shapes differ: the same kind of near-tie flip as in H21 is the likely cause, but speed_ab did not check gaps, so **unverified**. Decode cost does not depend on token values (64 tokens, no stops), so the throughput result stands | same |
| E20 | H5 per-phase placement | saves < 1% of a step; does not pay | fixed 85.63 s vs per-phase 86.20 s (load 0.29 s + free 0.19 s per step): **−0.57 s** | yes | **confirmed** (does not pay) | the reason differs from the prediction: not "prefetch hides training copies" but "training copies are not on the critical path at all" (H3) | same |
| E20 | D0 drafter sanity | base greedy mean committed tokens per pass ≥ 3 | **5.74** (96 prompts, batch 32; 16/16 accepted in 7.7% of row-passes); paper τ ≈ 6.5 on math | yes | **confirmed**: the re-implementation works; DFlash results are valid | — | `dflash_greedy_base*.json` |
| E20 | H21 greedy identity | main adapter, same-session plain run: ≥ 90% identical, every divergence at a top-2 gap < 0.05 | **main: not run (adapter missing).** Base, same session, batch 32: **82/96 (85.4%) identical**; all 14 divergences at gaps ≤ 0.031 (7 exact ties in fp16, the rest one fp16 step at that logit size). vs the saved evaluation (other session): 78/96, 18/18 near-ties. Greedy wall time: plain 936 s vs DFlash 352 s = **2.66×** (passes 1,536 → 322) | yes | as registered: **not tested yet**. On base: bug rule **passes** (no divergence at a clear gap), ≥ 90% share **fails** (85%) | the 90% was a guess at how often a 512-token answer meets an fp16 tie; 15% of answers do. Not a bug signal (the bug rule was the part meant to catch bugs) | same |
| E20 | H7 acceptance vs policy shift | base > control > main | base 5.74; control, main: not run | — | **not tested yet** | follow-up | — |
| E20 | H22 sampling exactness | main adapter, chi-square per position 1–15, all p > 0.05/15; biased rule min p < 0.05/15 | **base** (adapter missing): exact rule min p **0.244** (all 15 > 0.0033); accept-all min p **0** → test has power. Acceptance at T = 1: 3.60 | yes, for the base policy | **confirmed on base**; main: not tested yet | — | `dflash_sampling_base.json` |
| E20 | H23 speedup vs batch | x(8) > x(16) > x(32); 48 OOM-splits | T = 1, 512 tokens: **x3.42 (B 8) > x3.09 (B 16) > x1.71 (B 32)**; DFlash tok/s 45.5 / 72.8 / 78.7; acceptance 5.03 / 4.52 / 4.54. Batch 48: **crashed in the plain arm** (E19) | 8–32 yes | **confirmed** for 8/16/32; 48: not tested yet | E19 below; plain arm also ran with the drafter on the GPU (fixed: drafter moved to CPU for the plain arm) | `dflash_sweep.txt` |
| E20 | H6 real GRPO step | original: step ↓ ≥ 2×; updated: generation ↓ 1.2–1.8×, step ↓ 1.2–1.7×, H6 likely rejected | A-B-B-A, same seed/prompts: plain (gen batch 48) step **421.4 s**, generation 320.1 s; DFlash (2 × 24) step **431.7 s**, generation 331.0 s → **0.98× (2.4% slower)**. Passes 512 → 323 (≈ 161 per batch: ↓ 3.2×, inside 2.5–3.5); acceptance 4.61; verify 260 s (79% of generation, 0.80 s per pass vs 0.625 s plain), draft 45 s, accept 20 s | yes (repeats within 1.6 s) | **H6 rejected**; the updated expectation was also too optimistic (generation 0.97×, not 1.2–1.8×) | under the cap the drafter (1.06 GiB + cache) halves the batch. Plain L2L decode is copy-bound, so its throughput grows almost linearly with batch (B 48 ≈ 1.5× B 32), while a 16-token verify pass is no longer free (compute shows through: 0.80 vs 0.60 s). DFlash at B 24 ≈ plain at B 48. DFlash pays only when the batch is small (×3.4 at B 8) | `h6_steps.json`, `h6_*.txt` |


## E21 — follow-up notebook stopped: adapters not found (2026-11-27, soup_dflash2, git c9823b8)

The new guard cell stopped the run after ~3 minutes (tests 77/77, CUDA equivalence PASS):
"adapter(s) not attached: ['main', 'control']". Either the inputs were not added, or my search
could not see them: `Path.glob("**")` does not follow symbolic links (checked locally: a
symlinked input folder is invisible to it), and Kaggle input folders can be symlinks. The same
search ran in the first DFlash run, so its "adapters found: none" may have the same cause. Fix:
walk `/kaggle/input` with `os.walk(followlinks=True)` and print the files found, so a second miss
shows exactly what Kaggle mounted. No GPU results lost (the guard stopped before any).
Raw files: `results/kaggle/dflash2_try1/logs/`.
