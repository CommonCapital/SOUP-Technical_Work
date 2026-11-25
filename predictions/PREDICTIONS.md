# Pre-registered predictions

Written and committed **before** any of these experiments ran. The git commit
timestamp of this file is the proof. A rule changed after seeing numbers does
not count; if a rule must change, the change gets its own dated commit and the
report says so.

Method for every comparison: same instance, data, seed and warm-up; order
A-B-B-A; report median and 95% CI; a failed validity check makes the result
**unknown**, not "worse".

## Register A: memory budget (valid for this card, driver, PyTorch only)

| ID | Prediction | Confirmed if | If wrong |
|---|---|---|---|
| H11 | nvidia-smi peak per phase within 15% of the paper estimate (rollout ≈ 6.7 GB) | every phase within 15% | break down the gap: context, fragmentation, cuBLAS workspace, caching |
| H12 | Unchunked logits for 8k tokens OOM; 512-token chunks fit | unchunked fails and chunked stays < 8 GB | find the real largest transient with the memory profiler |
| H13 | CUDA context + workspace = 0.4–0.7 GB | nvidia-smi after init + one matmul, before weights, in range | update the budget; reduce workspace |
| H14 | Host RAM peak 13–17 GB | RSS + container limit logged every second, peak in range | spill checkpoints to disk or fewer rollouts per pass |

## Register B: time and bottleneck (never compare across hosts)

| ID | Prediction | Confirmed if | If wrong |
|---|---|---|---|
| H1 | Generation is 80–90% of step time | phase timers over 5 steps: generation > 70% | training pass is the bottleneck; optimise it first |
| H15 | Decode pass time within 20% of 7.26 GB ÷ measured GB/s | measured / predicted in [0.8, 1.2] | profile launches/sync with CUDA events |
| H16 | Copy < 10% of training time | profile copy vs compute streams | more tokens per pass or fix overlap |
| H3 | Prefetch hides > 80% of copy time in training, not in decoding | A/B prefetch on/off, A-B-B-A | check buffer reuse and stream sync |

## Register C: L2L correctness and placement (tolerance from a measured bf16 noise floor)

| ID | Prediction | Confirmed if | If wrong |
|---|---|---|---|
| H17 | L2L = plain model | fp32 max rel. error ≤ 1e-5 on logits and every LoRA grad; bf16 ≤ 1.5× noise between two plain runs | bisect by layer; all L2L results void until fixed |
| H18 | L2L KV decoding = plain decoding | greedy tokens identical on 50 prompts (fp32, tiny model) | check KV indexing and position IDs |
| H4 | Copies = 36 × passes, independent of batch | copy counter in engine log | find where layers are reloaded |
| H2 | In decoding, 0 resident layers + bigger batch beats 8 resident layers | A/B at equal memory, tokens/s higher for R=0 | adopt residency; explain why the model was wrong |
| H5 | Per-phase placement (10Cache/D2 idea) beats one fixed plan | lower total step time incl. setup, A/B over 5 steps each | keep one plan; report phase switching does not pay |

## Register D: data streaming and merge

| ID | Prediction | Confirmed if | If wrong |
|---|---|---|---|
| H19 | Saving (epoch, position, RNG) gives exact resume | kill -9 at random steps; reconciled ID log == uninterrupted log; 0 duplicates, 0 gaps | save state atomically after optimizer step |
| H20 | Tensor-by-tensor merge GPU peak < 1 GB | nvidia-smi during merge | find the tensor held too long |
| H27 | Merged = adapter | logit diff at bf16 noise level on 50 prompts; greedy agreement ≥ 99% | check α/r scale, module mapping, transpose |

## Register E: DFlash (drafter trained on base Qwen3-4B)

| ID | Prediction | Confirmed if | If wrong |
|---|---|---|---|
| H6 | Decoding passes ↓ 3–4×, step time ↓ ≥ 2× | A/B with/without drafter at the same step | show where time goes: drafter, verify compute, batching |
| H7 | Acceptance length drops between step 0 and step 40 | acceptance vs KL(policy‖base) logged at 0/10/20/40 | drift small; report LoRA barely moved the policy |
| H21 | Greedy output 100% identical with the current LoRA | compare with plain L2L decoding after training | verifier not using current weights, or acceptance wrong |
| H22 | Sampling keeps the policy's distribution | token-frequency test p > 0.05, 2,000 samples per method | fix acceptance rule; DFlash results void until then |
| H23 | Speedup lower at batch 64 than batch 8 | sweep batch 8 / 32 / 64 | report best batch with and without DFlash |

## Register F: did training really work (500 questions resolve ~4 points)

| ID | Prediction | Confirmed if | If wrong |
|---|---|---|---|
| H24 | Chosen difficulty gives base accuracy 20–60% | pilot 100 prompts × 4 samples, T = 1.0, per level (see note) | change difficulty or task before training |
| H25 | Zero-advantage groups < 50% per step | logged per step | raise temperature or filter prompts |
| H8 | GRPO improves held-out accuracy by +5 to +15 points | paired bootstrap (10k) on 500: 95% CI excludes 0 | **DON'T SHIP**; check lr, reward, signal share |
| H26 | Shuffled-reward control does not improve | CI of its gain includes 0 | gain comes from format, length or an eval bug; main gain void until explained |
| H9 | Gain is skill, not format | value-correct rate rises, not only tag-present; independent checker agrees | model is gaming: **DON'T SHIP** |
| H10 | Other skills not damaged | ~100 GSM8K, paired, within CI of base | lower lr or add KL; report trade-off |
| H28 | Adapter really moved the policy | KL(policy‖base) on held-out > control's, grows with steps | check adapter attachment and scale |

Note on H24: the plan's register says 200 prompts per difficulty, its pilot
section says 100 prompts × 4 samples. **Decision (made before the pilot):
100 prompts × 4 samples per level** (CI ≈ ±10 points, enough to place a level
in a band, ~19 min per level). If the chosen level's mean is within 10 points
of the 20% or 60% edge, add 100 more prompts for that level only.

## Clarifications added 2026-11-24 (before any GPU run)

- **H4 count.** "Copies = 36 × passes" is made precise as: each layer is
  copied at most once per direction per pass, and the count does not depend on
  batch size. A training pass is forward + backward, so 2 × 36, minus 1 when
  the last forward layer is still in the ring buffer at the turn (measured on
  CPU: 2L − 1).
- **H27 numeric rule.** "Logit difference at bf16 noise level" is made
  concrete: max |Δ log p| of merged vs adapter on the adapter's own greedy
  answers ≤ 3 × the noise floor (same adapter model, micro-batch 1 vs 8), and
  greedy token agreement ≥ 99%, and merged weights ≠ base weights.
- **Metric for H17.** Elementwise relative error is undefined near 0, so
  "max relative error" is computed as max|x − ref| / max|ref|; relative
  Frobenius error is reported too. bf16 passes only if both metrics are within
  1.5× of the measured eager↔sdpa noise.

## New hypothesis added 2026-11-24 after E07 (before any GPU run)

| ID | Hypothesis and why | Prediction | Test and decision rule | If wrong |
|---|---|---|---|---|
| H29 | The bf16 merged file keeps the learned change for the real run. Estimate: after ~40 Adam steps at lr 2e-5, B entries ~1e-4–8e-4, so ΔW elements ~1e-4–4e-4, while half a bf16 step of a typical Qwen3-4B weight (~0.02) is ~6e-5 (step 2⁻¹³ ≈ 1.2e-4) | `delta_lost_to_rounding_share` < 30% and merged-bf16 passes H27 | merge.py report + merge_check.py; confirmed if H27 passes for the bf16 file | the bf16 merged model is NOT the trained model: ship the adapter (or an fp32 merge, 2× size) and say so; report the lost share and the eval gap merged-bf16 vs adapter |

## Hardware change and H30, added 2026-11-24 (before any GPU run)

The GPU is a free Kaggle T4 (16 GB, Turing, no bf16), not an 8 GB Ampere
card. The 8 GB cap is enforced by a ballast process. All runs use fp16 weights
and matmuls with an fp32 residual stream. Time predictions (H1, H15, H16)
stay as written; any gap from the slower card is reported, not re-predicted.

| ID | Hypothesis and why | Prediction | Test and decision rule | If wrong |
|---|---|---|---|---|
| H30 | fp16 with an fp32 residual does not change Qwen3-4B's behaviour: fp16 has more mantissa bits than bf16, the weights are bf16 values (fp16 can hold them except tiny ones below ~6e-5, which become less precise), and the residual, where overflow would happen, stays fp32 | teacher-forced top-1 agreement with fp32 ≥ 99%, mean \|Δ log p\| ≤ 0.01, zero non-finite values (8 pilot prompts, 128 tokens) | `scripts/gpu/fp16_check.py`; all three must hold | do not pilot or train in fp16: find the layer that overflows; run that part in fp32 |

## Prompt v2 and H31, added 2026-11-24 (before any Qwen3-4B pilot)

Prompt v1 was changed after a **local smoke test on Qwen3-0.6B** (E09), not
after any 4B result: the small model copied the prompt's numeric example and
answered in ~20 tokens without working, often adding "* 1". v2 removes the
example, asks for brief visible work and restates "no other numbers".

| ID | Hypothesis and why | Prediction | Test and decision rule | If wrong |
|---|---|---|---|---|
| H31 | v2 gives the 4B model a fairer start: no example to copy, room to work before answering | v2 accuracy ≥ v1 at n4; v2 answers longer but truncation < 5% | Pilot both at n4 (100 × 4). Choice rule: if exactly one is in 20–60% → that one; if both → more mixed prompts; if neither → higher accuracy (then the search loop runs with it). Both reported | keep v1; report that the example did not hurt the 4B |

## Pilot level-selection rule (written before seeing any pilot number)

1. Levels tried in order: n4 (middle) → then n3 or n5 depending on result.
2. 0–5%: first read 20 answers by hand (checker? truncation? thinking on?); if
   genuinely failing, go easier (n3, then n3_addsub).
3. 90–100%: first check the reward is not too lax; then go harder (n5, n6).
4. Pick the level with mean success in 20–60% **and** the highest share of
   prompts with mixed results (1–3 of 4 correct). Tie → closer to 40%.
5. Stop after ≈ 5 pilots; if none fits, take the closest and report it.
6. Truncation must be < 5% at 256 new tokens, else raise the cap.

## Pilot 2, added 2026-11-25 (after pilot 1, before pilot 2 runs)

Pilot 1 (E01) chose **n4 with prompt v2** (31.2% after the E11 checker fix), but
65% of n4 answers were cut off at 256 tokens. Rule 6 above says: raise the cap.
The checker fix (E11) is a bug fix found by reading answers by hand; it only
turned wrong into right (54 answers, none the other way) and is applied to all
pilot numbers from now on.

**Cap choice: 512 new tokens.** Not higher, for the free-GPU budget: one decode
pass takes 0.61 s on the T4 (E12), so 512 tokens per GRPO step is ≈5 min of
generation per step and ≈4.5 h per 40-step run; 768 tokens would make the two
runs plus evaluation exceed the ~30 GPU-hours per week Kaggle gives.

| ID | Prediction | Test and decision rule | If wrong |
|---|---|---|---|
| H32 | At 512 tokens, n4/v2 accuracy rises to 40–55% and truncation falls to 10–35%; n5/v2 rises to 10–25% | Pilot 2: n4/v2 and n5/v2, 100 prompts × 4, T = 1.0, 512 new tokens, batch 48 | report; the selection rule below decides anyway |

**Selection rule for pilot 2** (replaces the pilot-1 choice once it has run):
1. Candidates: n4/v2 and n5/v2 at 512. n3 is not rerun: it was already above
   60% at 256 tokens (62.3%), and a longer cap can only finish more answers.
2. Pick the level in 20–60% with the highest mixed share (tie → closer to 40%).
   If neither is in range, take the one closest to 40% and report it.
3. Training and evaluation use **512 new tokens** whatever the truncation share.
   If truncation of the chosen level is still ≥ 5%, it is reported, and H9 is
   judged on two numbers instead of one: accuracy, and accuracy among answers
   that finished. A gain that comes only from fewer truncations (shorter
   answers) is a format gain, not a skill gain (see H9).
4. If the chosen level is within 10 points of 20% or 60%, 100 more prompts are
   drawn from that level's train split, and those IDs are removed from training.

## Pilot 2 outcome and the training setup, added 2026-11-25 (before the dry run)

Pilot 2 (E13): n4/v2 56.2% (mixed 44%), n5/v2 21.5% (mixed 47%). Rule 2 → **n5/v2**
(higher mixed share). n5 is within 10 points of 20% → rule 4: 100 more prompts
from `train_n5.jsonl` (its last 100 rows → `data/pilot_n5_extra.jsonl`;
training reads `data/train_n5_main.jsonl`, 1,400 rows, `scripts/split_extra.py`).
**If n5 over all 200 pilot prompts is below 20%, the choice moves to n4/v2**,
the only other level in range. Otherwise n5 stays.

**Training configuration (fixed now; main run and control differ only in `--shuffle-rewards`):**
level n5, prompt v2, 512 new tokens, T = 1.0, group 8, **6 prompts per step**
(48 rollouts), 40 steps, LoRA r16 α32 on the 7 projections, AdamW lr 2e-5,
warm-up 2 steps, β = 0 (KL to base logged every 5 steps), std-scaled
advantages, grad clip 1.0, fp16 + fp32 residual, seed 1234, micro-batch 8.

Deviation from the plan: **6 prompts per step instead of 16.** Reason: at 512
tokens only 48 sequences fit in one generation batch under 8 GB (pilot 2:
batch 48 peaked at 7.0 GiB allocated); 16 prompts = 128 rollouts = 3 batches
= ~16 min of decoding per step, ~12 h per run, two runs + evaluation >30 h of
free GPU. With 48 rollouts: one batch, ~5.2 min decoding per step.

| ID | Prediction (dry run, this configuration) | Confirmed if | If wrong |
|---|---|---|---|
| H11b | Peak of our processes ≤ 8,192 MiB in every phase (generate 48×512, train pass, Adam step) | `budget_check.py` our_peak_mib ≤ cap; trainer step logs | lower micro-batch / gen batch, re-measure |
| H1 | Generation is 70–90% of step time | dry run, 5 steps: generate / total | training pass is the bottleneck: optimise it first |
| H33 | Step time 5.5–8 min, so 40 steps = 3.7–5.3 h | dry run mean of steps 1–4 | re-plan the budget before the main run |
| H25 | Zero-advantage groups < 50% per step | dry run, logged per step (with G = 8, P(group all-same) is lower than the pilot's 4-sample share) | raise temperature or filter prompts |
| H19 | Resume after kill -9 is exact on the 4B | `resume_check.py`: IDs identical and no gaps (PASS_resume); completions identical is reported (PASS_exact) | fix before the main run; the main run must survive a Kaggle session restart |
| — | lr = 0 leaves the adapter at zero and KL at 0 | 2 steps, lr 0 | the optimiser or adapter wiring is wrong |

## Amendment after dry run 1 (2026-11-25, before dry run 2 and before any real training)

Dry run 1 (E14) ran the 31 GB host out of RAM in the training pass. Change:
**24 sequences per training pass** (`--train-seqs-per-pass 24`, two passes per
step) instead of 48. The update is mathematically the same (gradients
accumulate over both passes before one optimizer step; the loss normaliser is
the constant N_seq × max_new). Expected cost: +~2–5 s per step (one more weight
stream). New prediction: **host RAM peak ≤ 22 GB** (H14b); measured from the
per-second host log in dry run 2. Nothing else in the training configuration
changes. Dry run 2 repeats the 5 steps, the 4B resume test (now on easy puzzles
so updates really happen) and the lr = 0 check, then the 100 extra n5 pilot
prompts.
