# Discussion notes: why each choice was made

Plain-language prep for the 40-minute live discussion. One entry per decision,
added as the work happens. Each entry: what, why, what would prove it wrong.

## Day 1 (2026-11-23)

**Why Countdown.** Reward is exact and cheap; difficulty has a knob (3–6
numbers); fresh puzzles are unlimited so train and test never overlap; answers
are short (generation is the bottleneck). GSM8K: Qwen3-4B may already be ~90%.
JSON: risk of learning only the format.

**Why puzzles are built from an expression tree.** Every puzzle is guaranteed
solvable, and we store one reference solution. If the generator emitted
unsolvable puzzles, a 0 reward would sometimes be the puzzle's fault, not the
model's, and nobody would notice (silent failure).

**Why the checker uses `ast` and Fractions, never `eval`.** `eval` on model text
can run code. Fractions make 7/2*4 exactly 14, and stop 64.9999 from passing.
Only + − × ÷, parentheses and integer literals are allowed. No unary minus:
otherwise the model could negate without using a number.

**Why the last `<answer>` counts.** If any answer counted, the model would
learn to write several. With "last one counts", spraying answers never helps.

**Why `= 65` at the end is accepted.** Writing "55 + 36 − 19 − 7 = 65" is a
correct answer in a normal format. Rejecting it would punish correct answers
(silent failure "checker rejects a correct answer in another format"). The
right side must be an integer equal to the target.

**Why four sub-checks.** If accuracy rises but only "tag present" moves, the
model learned the format, not the skill. Logging all four separates the two.

**Why thinking mode is off.** Qwen3 would otherwise write long `<think>` blocks;
every token is a full pass over 7.26 GB of streamed weights (~0.6 s), so long
answers would blow the time budget. The template still inserts an empty think
block: that is how Qwen3 signals "thinking off".

**Why resume is designed this way.** The order within an epoch is a pure
function of (seed, epoch), so only (epoch, position, step) plus the sampling RNG
need saving. State is written atomically (temp file + rename) after the
optimizer step: a kill either sees the old state or the new one, never half.
The ID log is written before the work, so after a kill the last line may belong
to a step that never finished; resume drops it. Tested with 10 `kill -9`, one
inside the batch that crosses the epoch boundary.

## Day 2 (2026-11-24): the engine

**What L2L does, in one breath.** Weights sit in CPU RAM, one flat block per
layer. For each layer: copy it to the GPU once, push every sequence through
it, move on. A second CUDA stream copies layer N+1 while layer N computes,
using a ring of 3 GPU buffers, with events so a buffer is never overwritten
while still in use.

**Training pass = checkpointing at layer boundaries.** Forward without
gradients, saving only each layer's input (to CPU RAM). Then compute log-probs
only for answer tokens, 512 rows at a time (full logits for 8k tokens would be
~5 GB). Then go backward through the layers in reverse: reload the layer,
recompute it from its saved input with autograd on, push the gradient through.
Only LoRA A/B get gradients.

**Why compare against Hugging Face, not my own plain model.** Comparing the
engine against itself would hide a shared math bug. HF's Qwen3 is an
independent implementation, so matching it to 1e-6 in fp32 checks both the
Qwen3 math (q/k norm, RoPE, GQA) and the streaming.

**Why the bf16 tolerance is measured, not chosen.** bf16 is never bit-exact.
Running HF twice with two attention kernels (eager vs sdpa) shows how much
two correct implementations disagree (1.6% on logits here). The engine must
be within 1.5× of that. It was bit-identical to the sdpa path on logits.

**Two bugs the tests found.** (1) A scored token right after padding would be
predicted from a pad position. Fixed: a position is scored only if the token
before it is real. (2) My copy-count expectation was wrong: at the turn from
forward to backward, the last layer is still in a buffer, so it is reused.
The real claim is "never more than once per direction, independent of batch".

**Reference model for KL costs no PCIe.** The same loaded layer is run a
second time with the adapter switched off, so no extra copies.

**Why groups with zero advantage are skipped in training.** With β = 0 their
gradient is exactly zero, so streaming them through backward is wasted time.

**Constant loss normalizer.** Loss is divided by (number of answers × 256),
not by each answer's length, so long and short answers are weighted equally
per token (no length bias).

**Adam's first step surprise (0.6B on the Mac).** With lr 1e-4 and only one
useful group, one update changed the model's style from one-line answers to
long reasoning. Reason: Adam's first update is about lr × sign(gradient) for
every parameter, however small the gradient. So lr matters more than the
gradient size early on. On the GPU: lr 2e-5, and watch answer length.

**The merge that was mathematically right and still wrong (E07).** On the
0.6B test adapter, the merged bf16 model behaved exactly like the base model.
Debugging in two halves: merging in fp32 in memory matched the adapter to
2e-4, so the formula W + (α/r)·B·A, the names and the transpose are correct.
The problem is storage: bf16 has only 8 bits of mantissa, so a weight of 0.02
can only change in steps of about 0.00012 (2⁻¹³), and a change smaller than
half a step (0.00006) is rounded away. The adapter's changes were
smaller, so rounding threw away 88% of them. The adapter itself keeps working
because it is computed separately in fp32. Lesson for the interview: a check
that only asks "did the merge run and keep the same size" would have passed.
That's why the plan compares the merged model's *outputs* with the adapter's.
Whether it hits the real 4B run depends on how far training moves the
weights (H29).

**My own test harness had a bug (E06).** `kill -9 $!` after `uv run ...` kills
uv, not Python. The "killed" trainer kept running next to the resumed one.
Fixed, and the trainer now locks its run folder so this can't happen in a
real run either.

## Kaggle T4 (2026-11-24)

**Why fp16 needs an fp32 residual.** fp16 can't go above 65,504. Each layer
adds its output to a running "residual" vector, and in big models that vector
can grow past fp16's limit and become infinity. So the residual stays fp32 and
only the matrix multiplies (where the speed is) run in fp16. bf16 doesn't
have this problem (same range as fp32) but has fewer precision bits, which is
why fp16 mode came out 10× more precise on the tiny test.

**Why a ballast process and not a memory limit.** PyTorch's memory fraction
only limits PyTorch's own allocator inside our process; it doesn't count the
CUDA context and isn't what nvidia-smi shows. A second process that grabs
everything above 8 GB makes the card behave like an 8 GB card for us: if we
needed more, we'd crash, just as on real hardware.

**Why check fp16 before the pilot.** If fp16 quietly changed the model's
answers, the pilot's accuracy would describe a different model than the one
we train. So the notebook stops if H30 fails.

**Open point noticed.** The plan's register says 200 prompts per difficulty for
H24, the pilot section says 100 × 4. Decided before the pilot: 100 × 4, plus
100 more for the chosen level if it is within 10 points of an edge.
