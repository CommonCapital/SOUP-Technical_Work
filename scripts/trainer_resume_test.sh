#!/usr/bin/env bash
# H19 with the REAL trainer: run A uninterrupted; run B killed with kill -9 in
# the middle of step KILL_STEP, then resumed. Pass = identical sample IDs,
# rewards, completions and adapter norms per step.
# Usage: bash scripts/trainer_resume_test.sh <model> <dtype> [steps] [kill_step]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-Qwen/Qwen3-0.6B}; DTYPE=${2:-fp32}; STEPS=${3:-4}; KILL_STEP=${4:-3}
OUT=runs/resume_test; rm -rf "$OUT"; mkdir -p "$OUT"
ARGS=(--model "$MODEL" --dtype "$DTYPE" --data data/train_n3.jsonl --steps "$STEPS"
      --prompts-per-step 4 --group 4 --max-new 64 --gen-batch 16 --train-seqs-per-pass 16
      --micro-batch 4 --lr 1e-4 --log-kl-every 2)

# Call the venv's python directly: with `uv run`, $! would be the uv wrapper and
# kill -9 would leave the real trainer running (found the hard way, E06).
uv sync -q
PY=.venv/bin/python
"$PY" scripts/train_grpo.py "${ARGS[@]}" --run-dir "$OUT/a" > "$OUT/a.log" 2>&1

"$PY" scripts/train_grpo.py "${ARGS[@]}" --run-dir "$OUT/b" > "$OUT/b1.log" 2>&1 &
PID=$!
until grep -qs "\"step\": $KILL_STEP," "$OUT/b/state.json"; do sleep 0.2; done
sleep 3
kill -9 "$PID" || true
wait "$PID" 2>/dev/null || true
if kill -0 "$PID" 2>/dev/null || pgrep -f -- "--run-dir $OUT/b" >/dev/null; then
  echo "FAIL: trainer still alive after kill -9"; exit 1
fi
echo "killed during step $KILL_STEP: $(head -c 160 "$OUT/b/state.json")"
"$PY" scripts/train_grpo.py "${ARGS[@]}" --run-dir "$OUT/b" > "$OUT/b2.log" 2>&1
grep -E "resumed" "$OUT/b2.log"

uv run python - "$OUT" <<'EOF'
import json, sys, time
out = sys.argv[1]
L = lambda p: [json.loads(l) for l in open(p)]
a, b = L(f"{out}/a/ids.jsonl"), L(f"{out}/b/ids.jsonl")
ma, mb = L(f"{out}/a/metrics.jsonl"), L(f"{out}/b/metrics.jsonl")
sa, sb = L(f"{out}/a/samples.jsonl"), L(f"{out}/b/samples.jsonl")
res = {
  "ids_identical": [x["ids"] for x in a] == [x["ids"] for x in b],
  "steps_logged": [len(a), len(b)],
  "rewards": [[x["reward"] for x in ma], [x["reward"] for x in mb]],
  "adapter_norms": [[x["adapter_delta_norm"] for x in ma], [x["adapter_delta_norm"] for x in mb]],
  "samples_identical": [s["completion"] for s in sa] == [s["completion"] for s in sb],
  "metrics_steps_b": [x["step"] for x in mb],
}
res["PASS"] = (res["ids_identical"] and res["samples_identical"] and res["rewards"][0] == res["rewards"][1]
               and res["adapter_norms"][0] == res["adapter_norms"][1])
p = f"results/logs/trainer_resume_{time.strftime('%Y%m%d-%H%M%S')}.json"
open(p, "w").write(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=1)); print("log:", p)
EOF
