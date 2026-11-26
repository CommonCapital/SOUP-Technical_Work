# Qwen3-4B + GRPO + LoRA under an 8 GB GPU cap — **DON'T SHIP**

Every number below comes from a saved log in `results/` and is reproduced by
`uv run python scripts/final_analysis.py` (verification script). Rules and
predictions were committed to git *before* each run (`predictions/PREDICTIONS.md`,
commit times are the proof); the experiment log keeps every failure unedited
(`results/RESULTS_LOG.md`, entries E00–E17).

## 1. Verdict

GRPO training on 5-number Countdown puzzles **did** beat the base model on 500
held-out puzzles (+5.6 points). But the **shuffled-reward control, trained with
no real signal, gained more** (+9.6). So the gain cannot be credited to the
reward. Both gains come mostly from answers that **finish within the 512-token cap**
rather than better arithmetic: among finished answers, accuracy *fell*. Two of the
pre-registered SHIP conditions fail (H26 control, H9a skill vs format), so the
verdict is **DON'T SHIP**. The system itself (own layer-to-layer engine, 8 GB cap,
exact resume, two independent checkers) works and is verified.

| Model (held-out n5, 500, greedy, 512 tok) | Accuracy | Paired Δ vs base [95% CI] | McNemar p | Finished | Acc. among finished |
|---|---|---|---|---|---|
| Base Qwen3-4B | 19.2% | — | — | 21.2% | 90.6% (106) |
| Base + adapter (main, 40 steps) | 24.8% | **+5.6 [+1.2, +10.0]** | 0.017 | 29.6% | 83.8% (148) |
| Control (shuffled rewards, 40 steps) | 28.8% | **+9.6 [+5.4, +13.8]** | 1e-5 | 33.6% | 85.7% (168) |
| Main − control | | −4.0 [−8.4, +0.4] | 0.10 | | |
| Merged into weights, stored bf16 | 26.2% | +7.0 [+2.8, +11.4] | 0.002 | 29.2% | 89.7% (146) |
| GSM8K, 100 questions: base → main | 94% → 92% | −2 [−6, +2] | 0.63 | | |

**What must change before this could ship:** (1) remove the length confound: re-pilot
with a cap or prompt where < 5% of base answers are cut off (my own rule 6; I relaxed
it to 512 tokens for the free-GPU budget, and that is what the verdict turned on);
(2) ≥ 3 seeds per arm, decided on the mean (one control seed can drift lucky);
(3) ship the **adapter** or an fp32 merge: the bf16 merge lost 87% of the change (H29).

## 2. Setup

**Task.** Countdown: use every given number once with + − × ÷ to hit a target.
Puzzles are generated with a known solution (all 10,500 reference solutions score 1.0),
train/held-out/pilot splits share no puzzle (hash check). **Reward** 1/0 from an
AST-whitelisted exact-fraction checker (last `<answer>` block; cheat tests for
extra/repeated numbers, unary minus, floats, `**`, `//`, ×1 tricks).
**Pilot path** (rule written before each pilot): n4 with prompt v1 13.2% → prompt v2
(no example to copy, brief working) 31.2% but 65% cut off at 256 tokens → pilot 2 at
512 tokens: n4 56.2%, **n5 21.5%** (mixed groups 47%) → n5 chosen, confirmed 23.5%
over 200 prompts. **Training:** G = 8, 6 prompts/step (48 rollouts), 40 steps,
T = 1.0, LoRA r16 α32 on 7 projections (33.0 M params), AdamW lr 2e-5, 2 warm-up
steps, β = 0 (KL to base logged), std-scaled advantages, constant loss normaliser.
Control = identical run, same seed and prompt order, rewards shuffled.

**Hardware deviation.** The only GPU available was a free **Kaggle T4** (Turing, 16 GB,
no bf16, no SSH, ~12 h sessions, ~30 GPU-h/week). Handled as follows; total use ≈ 18 GPU-h.

| Difference | How it was handled | Evidence |
|---|---|---|
| No bf16 | fp16 weights + matmuls, **fp32 residual stream**; non-finite values counted, never sampled | H30 on real 4B: top-1 agreement 99.8%, mean \|Δlog p\| 0.0008, 0 non-finite; 2× faster than fp32 |
| 16 GB, not 8 | a **ballast process** holds all memory above 8,192 MiB, so our processes physically cannot exceed it (CUDA context included) | nvidia-smi every second, whole session; our peak **8,158 MiB** in every run |
| No SSH, session caps | self-contained notebooks (code + data embedded, hashes checked); exact resume | kill -9 test on the 4B (H19) |

## 3. Engineering

**Own L2L engine** (`src/soupgrpo/l2l.py`): weights live in pinned host memory, one flat
buffer per layer; a ring of 3 GPU buffers is filled by a copy stream with CUDA events
(prefetch); one engine does prefill, KV-cached decoding and training; training saves
each layer's input to host and recomputes in backward; log-probs in 512-row chunks.
**Equivalence:** vs HF Qwen3 on CPU, fp32 logits 1.1e-6 and all LoRA grads ≤ 2.1e-6,
bf16 within the measured noise floor, greedy 50/50 identical (H17, H18); on CUDA with
prefetch (2 and 3 buffers, checksums on) identical to the CPU engine (E02b).

| Measured on the T4 | Predicted | Measured | Verdict |
|---|---|---|---|
| Decode pass (all 36 layers) | 7.26 GB ÷ 12.3 GB/s = 0.59 s | 0.61 s | H15 ✓ |
| Generation share of step time | 80–90% | 87% (main) | H1 ✓ |
| Step time / 40 steps | 5.5–8 min | 6.1 min / 4.04 h | H33 ✓ |
| GPU peak, generate 48×512 / train | ≤ 8,192 MiB | 7.17 / 4.05 GiB allocated; 8,158 MiB smi | H11b ✓ |
| CUDA context + workspace | 0.4–0.7 GB | 0.15 GiB | H13 ✗ (lower) |
| Unchunked logits, 8k rows | OOM | fits (6.7 GiB) | H12 ✗ |
| Host RAM | 13–17 GB | **30.9 GB → crash**, fixed → 22 GB | H14 ✗ |
| Merge GPU peak | < 1 GB | 0.57 GB | H20 ✓ |

**Resume (H19):** saving (epoch, position, RNG, adapter, optimizer) atomically gives
identical IDs/rewards/answers after kill -9 (Mac fake trainer: 10 kills/200 steps;
0.6B real trainer; 4B on the T4 with a real update). Not run: prefetch on/off and
residency A/B (H2, H3, H5) and DFlash (H6, H7, H21–H23): the GPU budget went to the
pilots and both full arms.

## 4. Did the model learn? (proof and its limits)

* **Signal reached the adapter:** adapter norm 0 → 0.76; exact KL(adapter‖base) on its
  own held-out answers **0.065/token for main vs 0.029 for control** (H28 ✓).
* **But not more correct answers than no signal:** main − control −4.0 [−8.4, +0.4] (H26 ✗).
  Training curves do not prove learning here: with identical prompt order, training reward
  rose the same in both arms (main +0.18, control +0.22 over 40 steps). I first read the
  main curve as learning (E16) and corrected it after the control (E17).
* **Format vs skill:** the value-correct rate rises with the reward, and the tag rate
  rises the same (+8.4): the extra correct answers are the extra *finished* ones (H9a ✗).
* **Checker is right:** an independent re-implementation (string search + hand-written
  parser, no shared code) agrees on 100% of 3,600 saved answers; on the pre-fix pilot files it
  flags exactly the 52 answers the E11 fix changed there. Hand audit of 30 trained answers: 30/30
  correctly scored, including one cheat attempt `(80/11)*(11/80)` (rejected).
* **Other skills:** GSM8K −2 points, not significant (H10 ✓).
* **Not detected by this setup:** overfitting to the puzzle generator's style (train and
  test share it); effects that need > 40 steps; seed-to-seed variance (one seed per arm).

## 5. Silent failures: what was checked

| Failure | Check | Result |
|---|---|---|
| Checker rejects correct answers | reading pilot answers by hand | **found (E11):** "write it inside the `<answer>` tags:" read as the answer; fixed (also "target = equation" order), 3 tests, 1,600 answers rescored (56 wrong→right, 0 right→wrong) |
| Checker accepts wrong answers | cheat tests, 2nd checker, audit of 30 | none found |
| Merge silently loses the change | merged ≠ base checksum + greedy/log-prob vs adapter | **found (E07, E16):** bf16 lost 87% of ΔW; check FAIL → ship adapter |
| Host RAM, not GPU, runs out | per-second host log | **found (E14):** buffer growth held 3 copies; fixed, reserved once |
| Mechanics test passes trivially | require a real update in resume / lr = 0 checks | **found (E14):** all rewards 0 → nothing tested; moved to easy puzzles |
| My own cap formula | recompute from raw nvidia-smi | **found (E12):** +107 MiB baseline not subtracted; fixed |
| Prefetch race, KV misalignment | CUDA vs CPU equivalence, greedy vs HF | none |
| Thinking mode on, wrong template | assert no `<think>` in outputs | 0% in all runs |
| Data overlap / unsolvable puzzles | hashes; reference solutions | 0 overlap; all solvable |
| Resume repeats or skips data | ID logs after kill -9 | identical |
| fp16 overflow | non-finite counter in every run | 0 |

## 6. Hypotheses: prediction vs result

✓ confirmed: H1, H4 (36 copies per pass, independent of batch), H8, H10, H15, H17, H18,
H19, H20, H24 (n5 23.5%), H25 (mean 46% zero-advantage groups, single steps up to 100%),
H28, H30, H33. ✗ rejected: **H26, H9(a)** (the verdict), H12, H13, H14, H27 (bf16), **H29**
(predicted < 30% of ΔW lost, measured 87%: I estimated ΔW at 1e-4–4e-4, real ΔW was mostly
below half a bf16 step ≈ 6e-5), H31 truncation part, H32 n4 part (56.2 vs ≤ 55). Not run:
H2, H3, H5, H6, H7, H16 (estimated 3–5% from copy counts, not profiled), H21–H23.

## 7. What surprised me, what still concerns me

The control result surprised me most: a run with shuffled rewards beat the real run. My
best reading is that 79% of base answers were cut off, so almost any change in verbosity
moves the score, and the metric rewarded finishing more than reasoning. I set the < 5%
truncation rule myself and then relaxed it for the GPU budget; the verdict turned on
exactly that relaxation. It still concerns me that I cannot separate "control drifted
lucky" from "the process itself shortens answers" with one seed per arm, and that the
bf16 merge, the normal way to ship, silently produced a different model.

## 8. AI tool use

The plan, code, notebooks and analysis were written with an AI coding assistant (Claude
Code); I ran every GPU notebook, read the outputs and made the calls on what to run next.
What was verified rather than trusted: the engine against Hugging Face's reference (fp32
1e-6), every checker change against hand-read answers and a second independent checker,
every memory and time claim against raw nvidia-smi and host logs. What was changed or
redone after checking: the checker (E11), the cap formula (E12), the host-memory buffer
and the trivially-passing mechanics tests (E14), and the assistant's own reading of the
training curve as learning, which the control disproved (E16 → E17).
