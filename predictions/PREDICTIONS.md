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

## Pilot level-selection rule (written before seeing any pilot number)

1. Levels tried in order: n4 (middle) → then n3 or n5 depending on result.
2. 0–5%: first read 20 answers by hand (checker? truncation? thinking on?); if
   genuinely failing, go easier (n3, then n3_addsub).
3. 90–100%: first check the reward is not too lax; then go harder (n5, n6).
4. Pick the level with mean success in 20–60% **and** the highest share of
   prompts with mixed results (1–3 of 4 correct). Tie → closer to 40%.
5. Stop after ≈ 5 pilots; if none fits, take the closest and report it.
6. Truncation must be < 5% at 256 new tokens, else raise the cap.
