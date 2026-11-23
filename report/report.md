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
