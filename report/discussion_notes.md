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

**Open point noticed.** The plan's register says 200 prompts per difficulty for
H24, the pilot section says 100 × 4. Decided before the pilot: 100 × 4, plus
100 more for the chosen level if it is within 10 points of an edge.
