# Qwen3-4B + GRPO + LoRA under an 8 GB GPU cap — **DON'T SHIP**

All numbers here come from saved logs in `results/`. You can reproduce them with
`uv run python scripts/final_analysis.py`. I wrote my predictions and decision rules into
`predictions/PREDICTIONS.md` and committed them *before* each run, so the git timestamps
show I did not change them afterwards. Every failure is kept as it happened in
`results/RESULTS_LOG.md` (entries E00–E20).

## 1. Verdict

The trained model is better than the base model on 500 new puzzles: **+5.6 points**.
But my control run, trained with shuffled rewards (so no real learning signal), got
**+9.6**, even more. So I cannot say the reward caused the gain. When I looked closer,
most of the gain in both runs comes from answers that now **finish inside the 512-token
limit**, not from better arithmetic. Among answers that do finish, accuracy actually went
*down*. Two of my own SHIP conditions fail (H26: beat the control; H9a: skill, not
format), so the answer is **DON'T SHIP**. The system around the model (my own
layer-by-layer engine, the 8 GB cap, exact resume, two independent checkers) works, and I
verified it.

| Model (500 held-out 5-number puzzles, greedy, 512 tokens) | Accuracy | Change vs base [95% CI] | McNemar p | Finished | Correct among finished |
|---|---|---|---|---|---|
| Base Qwen3-4B | 19.2% | — | — | 21.2% | 90.6% (106) |
| Base + adapter (main, 40 steps) | 24.8% | **+5.6 [+1.2, +10.0]** | 0.017 | 29.6% | 83.8% (148) |
| Control (shuffled rewards, 40 steps) | 28.8% | **+9.6 [+5.4, +13.8]** | 1e-5 | 33.6% | 85.7% (168) |
| Main − control | | −4.0 [−8.4, +0.4] | 0.10 | | |
| Adapter merged into weights, saved in bf16 | 26.2% | +7.0 [+2.8, +11.4] | 0.002 | 29.2% | 89.7% (146) |
| GSM8K, 100 questions: base → main | 94% → 92% | −2 [−6, +2] | 0.63 | | |

**What I would change before this could ship:** (1) Fix the length problem first. Pick a
token limit or prompt where fewer than 5% of base answers get cut off. That was my own
rule; I relaxed it to 512 tokens to fit the free GPU budget, and the verdict turned on
exactly that. (2) Run at least 3 seeds per arm and decide on the mean, because one control
seed can just get lucky. (3) Ship the **adapter**, or merge in fp32. The bf16 merge lost
87% of the change (H29).

## 2. Setup

**Task.** Countdown: use every given number once with + − × ÷ to reach the target. I
generate each puzzle from a known solution, so all of them are solvable (all 10,500
reference solutions score 1.0). Train, held-out and pilot sets share no puzzle (checked by
hash). **Reward** is 1 or 0 from a checker that only accepts a whitelist of Python syntax
and computes with exact fractions. It reads the last `<answer>` block and has tests for
cheats (extra or repeated numbers, unary minus, floats, `**`, `//`, ×1 tricks).
**How I picked the difficulty** (rule written before each pilot): 4 numbers with prompt v1
gave 13.2%. Prompt v2 (no example to copy, brief working) gave 31.2%, but 65% of answers
were cut off at 256 tokens. At 512 tokens: 4 numbers 56.2%, **5 numbers 21.5%**, with 47%
mixed groups. I chose 5 numbers, and it held at 23.5% over 200 prompts. **Training:** 8
answers per prompt, 6 prompts per step (48 rollouts), 40 steps, temperature 1.0, LoRA rank
16, α 32 on 7 projections (33.0 M parameters), AdamW lr 2e-5, 2 warm-up steps, no KL
penalty (KL to base is only logged), std-scaled advantages, constant loss normaliser. The
control is the same run with the same seed and prompt order, but with shuffled rewards.

**Hardware change.** The only GPU I had was a free **Kaggle T4**: Turing, 16 GB, no bf16,
no SSH, sessions of about 12 h, about 30 GPU-hours a week. This is how I handled it. In
total I used about 21 GPU-hours.

| Difference | What I did | Evidence |
|---|---|---|
| No bf16 | fp16 weights and matmuls, but an **fp32 residual stream**; non-finite values are counted and never sampled | H30 on the real 4B: top-1 agreement 99.8%, mean \|Δlog p\| 0.0008, 0 non-finite; 2× faster than fp32 |
| 16 GB, not 8 | a **ballast process** takes all memory above 8,192 MiB, so my processes physically cannot go over the cap (CUDA context included) | nvidia-smi every second for the whole session; my peak was **8,158 MiB** in every training run |
| No SSH, time limits | self-contained notebooks (code and data embedded, hashes checked) and exact resume | kill -9 test on the 4B (H19) |

## 3. Engineering

**My own L2L engine** (`src/soupgrpo/l2l.py`). The weights stay in pinned CPU memory, one
flat buffer per layer. A ring of 3 GPU buffers is filled on a separate copy stream with
CUDA events, so the next layer loads while the current one computes (prefetch). The same
engine does prefill, KV-cached decoding and training. In training it saves each layer's
input to CPU memory and recomputes the layer in the backward pass. Log-probs are computed
in chunks of 512 rows. **Is it correct?** Against Hugging Face's Qwen3 on CPU: fp32 logits
differ by 1.1e-6 and all LoRA gradients by ≤ 2.1e-6, bf16 is within the measured noise,
and greedy output is identical on 50/50 prompts (H17, H18). On CUDA with prefetch (2 and 3
buffers, checksums on) it matches the CPU engine exactly (E02b).

| Measured on the T4 | Predicted | Measured | Verdict |
|---|---|---|---|
| Decode pass (all 36 layers) | 7.26 GB ÷ 12.3 GB/s = 0.59 s | 0.61 s | H15 ✓ |
| Generation share of step time | 80–90% | 87% (main) | H1 ✓ |
| Step time / 40 steps | 5.5–8 min | 6.1 min / 4.04 h | H33 ✓ |
| GPU peak, generate 48×512 / train | ≤ 8,192 MiB | 7.17 / 4.05 GiB allocated; 8,158 MiB smi | H11b ✓ |
| CUDA context + workspace | 0.4–0.7 GB | 0.15 GiB | H13 ✗ (lower) |
| Logits without chunking, 8k rows | OOM | fits (6.7 GiB) | H12 ✗ |
| Host RAM | 13–17 GB | **30.9 GB → crash**, fixed → 22 GB | H14 ✗ |
| Merge GPU peak | < 1 GB | 0.57 GB | H20 ✓ |

**Resume (H19):** I save epoch, position, RNG state, adapter and optimizer atomically. After
kill -9 the run continues with identical prompt IDs, rewards and answers. Tested with a fake
trainer on the Mac (10 kills over 200 steps), the real trainer on 0.6B, and the 4B on the T4
with a real update.

**Speed tests and DFlash (bonus; one extra T4 session, arms run A-B-B-A inside it,
E19–E20).** I wrote my own DFlash implementation for the z-lab Qwen3-4B drafter. I only use
their weights: 5 layers that read target layers 1/9/17/25/33, KV injection, blocks of 16.
On CPU I tested it against plain greedy with random, "oracle" and half-oracle drafters.
That test found a NaN bug that would have been silent in greedy mode (E18). The batch sweep
found a second bug: when a batch ran out of memory, my fallback kept the failed batch's KV
cache alive through the exception, so it could never recover (E19; no earlier run hit it).

| Measured (T4, 8 GB cap) | Predicted | Measured | Verdict |
|---|---|---|---|
| Prefetch in decoding: share of copy time hidden | < 20%; pass ≈ copy time | 31%; 0.604 s vs copy 0.588 s (off: 0.785 s) | H3 ✗ / ✓ |
| Prefetch in the training phase | > 80% hidden | 7% (0.16 s of 85.6 s; copies are only 2.35 s) | H3 ✗ |
| Same memory: 0 layers kept + batch 48 vs 8 layers kept + batch 32 | 1.10–1.25× | 1.24× (79.5 vs 63.9 tok/s) | H2 ✓ |
| Keep 16 layers on the GPU during training only | saves < 1% | 0.57 s slower per step (load + free cost more) | H5 ✓ |
| Drafter acceptance, base model, greedy | ≥ 3 tokens per pass | 5.74 (paper ≈ 6.5) | D0 ✓ |
| DFlash speedup, T = 1, 512 tokens, batch 8 / 16 / 32 | falls as batch grows | ×3.42 / ×3.09 / ×1.71 | H23 ✓ |
| Real GRPO step: plain batch 48 vs DFlash 2 × 24 | ≥ 2× faster | 421 s vs 432 s (0.98×) | H6 ✗ |

The main lesson: under the cap, the drafter's ~1.5 GB halves the rollout batch. Plain L2L
decoding is limited by copying layers in, so it gets almost linearly faster with a bigger
batch. DFlash at 24 rows ends up equal to plain at 48: it needs 3.2× fewer passes, but each
16-token verify pass costs 0.80 s instead of 0.60 s. So DFlash only pays at small batch
sizes. Greedy identity on the base model (H21): 82/96 answers identical, and every
difference starts at an fp16 near-tie (top-2 gap ≤ 0.031), but that misses my 90% target.
Sampling stays unbiased (H22: all 15 chi-square p > 0.05/15, lowest 0.24, and the test does
catch a deliberately biased rule, p ≈ 0). The prefetch result in training suggests that
phase is limited by CPU-side work, but I have not profiled it. The trained adapters were
not attached to that session, so H7, and H21/H22 with the adapter, are still open.

## 4. Did the model learn? (proof and its limits)

* **The signal reached the adapter:** adapter norm 0 → 0.76. Exact KL(adapter‖base) on its
  own held-out answers is **0.065 per token for main vs 0.029 for control** (H28 ✓).
* **But it did not get more answers right than training with no signal:** main − control
  −4.0 [−8.4, +0.4] (H26 ✗). The training curves don't prove learning here either: with the
  same prompt order, the training reward rose by the same amount in both runs (main +0.18,
  control +0.22 over 40 steps). At first I read the main curve as learning (E16). The
  control showed I was wrong (E17).
* **Format vs skill:** the share of correct values rises with the reward, and the share of
  proper tags rises the same (+8.4). The extra correct answers are just the extra *finished*
  ones (H9a ✗).
* **The checker is right:** a second, independent checker (string search plus a
  hand-written parser, no shared code) agrees on 100% of 3,600 saved answers. On the old
  pilot files it flags exactly the 52 answers that the E11 fix changed. I also checked 30
  trained answers by hand: all 30 scored correctly, including one cheat attempt
  `(80/11)*(11/80)` (rejected).
* **Other skills:** GSM8K −2 points, not significant (H10 ✓).
* **What this setup cannot see:** overfitting to the style of my puzzle generator (train and
  test share it), effects that need more than 40 steps, and seed-to-seed variance (one seed
  per arm).

## 5. Silent failures: what I checked

| Failure | Check | Result |
|---|---|---|
| Checker rejects correct answers | reading pilot answers by hand | **found (E11):** "write it inside the `<answer>` tags:" was read as the answer; fixed (also the "target = equation" order), 3 tests, 1,600 answers rescored (56 wrong→right, 0 right→wrong) |
| Checker accepts wrong answers | cheat tests, second checker, audit of 30 | none found |
| Merge quietly loses the change | merged ≠ base checksum + greedy/log-prob vs adapter | **found (E07, E16):** bf16 lost 87% of ΔW; check FAIL → ship the adapter |
| Host RAM runs out, not GPU | host memory logged every second | **found (E14):** growing a buffer held 3 copies; fixed, reserved once |
| Mechanics test passes for a trivial reason | resume / lr = 0 checks must include a real update | **found (E14):** all rewards were 0, so nothing was tested; moved to easy puzzles |
| My own cap formula | recomputed from raw nvidia-smi | **found (E12):** +107 MiB baseline not subtracted; fixed |
| Prefetch race, KV misalignment | CUDA vs CPU equivalence, greedy vs HF | none |
| Speculative decoding reads unwritten KV | test with NaN-filled memory and rows drifting apart | **found (E18):** masked garbage → NaN (silent in greedy); zero-init |
| OOM fallback never recovers | Kaggle sweep at batch 48 | **found (E19):** retry ran inside `except`, failed KV kept alive; fixed + test |
| Thinking mode on, wrong template | assert no `<think>` in outputs | 0% in all runs |
| Data overlap / unsolvable puzzles | hashes; reference solutions | 0 overlap; all solvable |
| Resume repeats or skips data | ID logs after kill -9 | identical |
| fp16 overflow | non-finite counter in every run | 0 |

## 6. Hypotheses: prediction vs result

✓ confirmed: H1, H4 (36 copies per pass, same for any batch), H8, H10, H15, H17, H18, H19,
H20, H24 (n5 23.5%), H25 (on average 46% of groups had zero advantage; single steps up to
100%), H28, H30, H33, H2, H5, D0, H23 (batch 8–32), H22 (base). ✗ rejected: **H26, H9(a)**
(the verdict), H12, H13, H14, H27 (bf16), **H29** (I predicted < 30% of ΔW lost and measured
87%: I estimated ΔW at 1e-4–4e-4, but most of the real ΔW was below half a bf16 step, ≈ 6e-5),
the truncation part of H31, the n4 part of H32 (56.2 vs ≤ 55), H3 (decode compute is about
5.5 ms per layer, not 1–2; in training the copies are not on the critical path), **H6**
(DFlash 0.98× on a real step). Still open: H7, H21/H22 with the adapter, H23 at batch 48
(follow-up notebook is ready), H16 (estimated at 3–5% from copy counts, not profiled).

## 7. What surprised me, what still worries me

The control surprised me most. A run with shuffled rewards beat the real run. My best
explanation is that 79% of base answers were cut off, so almost any change in how long the
model talks moves the score, and the metric rewarded finishing more than reasoning. I set
the < 5% truncation rule myself and then relaxed it to fit the GPU budget, and the verdict
turned on exactly that. What still worries me: with one seed per arm, I can't tell "the
control got lucky" apart from "this training process itself makes answers shorter". And the
bf16 merge, the normal way to ship, quietly produced a different model.

## 8. AI tool use

**What was mine.** Before any code existed, I wrote the 28-page technical plan: the choice of
Countdown, the pilot funnel and its 20–60% rule, the hypotheses H1–H33 with numeric
predictions, the SHIP / DON'T SHIP rules, the shuffled-reward control, the silent-failure
list, and the DFlash and speed tests. Every decision in this report follows rules I wrote
there in advance. When the planned 8 GB GPU turned out to be a free Kaggle T4, I chose to go
ahead on Kaggle with an enforced 8 GB cap rather than stop. I insisted on checking the base
model's success rate before any training. I ran every GPU session myself and read the results.

**What the AI did.** An AI coding assistant (Claude Code) wrote the code and notebooks from
my plan, ran the local tests, did most of the analysis, and drafted this report. I did not take its output on trust. The engine is checked against Hugging Face's
reference (fp32, 1e-6), every checker change against answers read by hand and against a
second independent checker, and every memory and time claim against raw nvidia-smi and host
logs. Things changed or redone after checking: the checker (E11), the cap formula (E12), the
host-memory buffer and the mechanics tests that passed for a trivial reason (E14), two DFlash
bugs (E18, E19), and the assistant's own reading of the training curve as learning, which the
control disproved (E16 → E17).
