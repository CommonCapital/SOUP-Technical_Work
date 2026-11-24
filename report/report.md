# Qwen3-4B + GRPO + LoRA under an 8 GB GPU cap: SHIP / DON'T SHIP

*Draft skeleton. Filled in as results arrive; every number links to a raw log.
Limit: 3 pages.*

## 1. Verdict
**[SHIP / DON'T SHIP]**: one paragraph. Gain on 500 held-out Countdown puzzles,
paired 95% CI, control result, capability check, and the one or two limits that
matter most. If DON'T SHIP: the exact list of what must change first.

| Model | Held-out accuracy (500) | Paired Δ vs base [95% CI] | McNemar p |
|---|---|---|---|
| Base Qwen3-4B | | — | — |
| Base + adapter | | | |
| Merged | | | |
| Control (shuffled rewards) | | | |

## 2. Setup
Task and why (pilot path, including rejected levels), reward and its cheat
tests, data sizes, GPU/host/PCIe as measured, hyperparameters.

### 2.1 Hardware deviation: Kaggle T4 instead of an 8 GB Ampere card
No rented GPU was available, so every GPU result comes from a free Kaggle T4.
The task (and my own plan) assumes an 8 GB Ampere card. Each difference, how
it was handled, and the evidence:

| Difference | Risk | How it was solved | Evidence |
|---|---|---|---|
| T4 is Turing (sm_75): **no bf16** | bf16 is emulated/slow; Soup's L2L rejects fp16 | Engine runs **fp16 weights and matmuls with the residual stream kept in fp32** (fp16 max is 65,504; residual streams can exceed it). Non-finite values are counted and never sampled from. Training stops on a non-finite log-prob | H30: fp16 vs fp32 on real Qwen3-4B (top-1 agreement, Δ log p); tiny-model test: fp16 mode is ~10× closer to fp32 than plain bf16 |
| T4 has **16 GB**, not 8 | the 8 GB cap is not enforced by hardware | A **ballast process** holds all memory above 8,192 MiB for the whole session, so the trainer physically cannot use more (CUDA context included), like a real 8 GB card. An allocator limit would not count | `ballast.json`, per-second nvidia-smi log; our usage = total used − ballast, peak reported per run |
| **No SSH, capped sessions** (~12 h each, ~30 GPU h/week) | long runs cut off | Self-contained notebooks; runs resume exactly from the last committed step (tested with kill -9, H19) | trainer resume logs |
| Older PCIe host, slower tensor cores | different speeds than an RTX 3070 | Times are reported for this host only and never compared across hosts (plan rule) | `pcie_*.json`, phase timers |
| fp16 instead of bf16 changes the numbers | equivalence tolerance | Noise floor measured, not assumed; the fp16 tolerance comes from the same rule (≤ the plain-bf16 error vs fp32) | equivalence logs |

What this does not cover: results are for fp16 on a T4. bf16 on an Ampere
card would likely give the same accuracy but different step times.

## 3. Engineering
- L2L engine: design in 5 lines; equivalence test result (H17, H18).
- Memory: estimate vs nvidia-smi per phase (H11–H14), table.
- Time: phase split, PCIe-bound decode (H1, H15, H16, H3), residency A/B (H2).
- Data streaming + resume (H19); merge peak and merge = adapter (H20, H27).
- DFlash (H6, H7, H21–H23), or measured reason it did not help.

## 4. Proof the model learned
Sub-checks (format vs skill), independent checker, harder split, adapter norm
and KL vs control, merged = adapter, GSM8K, behaviour drift, manual audit of 30.
What this does **not** detect.

## 5. Hypotheses: prediction vs result
Compact table H1–H28: prediction, result, verdict, gap explanation.

## 6. Silent failures
Table: stage, failure, how checked, evidence, caught by Soup checks?

## 7. Limits: expected vs measured
Filled potential-vs-real table; one sentence for every row that differs.

## 8. What surprised me, what still concerns me
One paragraph (written by the author).

## 9. AI tool use
What was accepted, verified, changed or redone.
