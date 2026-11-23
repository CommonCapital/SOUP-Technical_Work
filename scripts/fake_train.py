"""Stand-in for the GRPO trainer loop, used to test kill -9 resume (H19).

Same order of operations as the real trainer will use:
  1. take batch from the stream          (cursor advances in memory)
  2. append attempted IDs to the log
  3. "rollouts + update"                  (sleep; draws from the sampling RNG)
  4. save state atomically, step += 1     (commit point)
On start, if a state file exists: load it, reconcile the log, continue.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from soupgrpo.data_stream import (PromptStream, append_id_log, load_state,  # noqa: E402
                                  reconcile_id_log, save_state)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sleep", type=float, default=0.02)
    a = ap.parse_args()

    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    state_path, log_path = run / "state.json", run / "ids.jsonl"

    loaded = load_state(state_path)
    if loaded:
        st, extra = loaded
        dropped = reconcile_id_log(log_path, st.step)
        rng = np.random.default_rng()
        rng.bit_generator.state = extra["sampling_rng"]
        print(f"resumed at step {st.step} (epoch {st.epoch}, pos {st.position}); dropped {dropped} uncommitted log lines",
              flush=True)
    else:
        st, rng = None, np.random.default_rng(a.seed + 1)

    stream = PromptStream(a.data, seed=a.seed, state=st)
    while stream.state.step < a.steps:
        step = stream.state.step
        batch = stream.next_batch(a.batch)
        draw = int(rng.integers(0, 2**31))            # stands in for rollout sampling
        append_id_log(log_path, step, [r["id"] for r in batch],
                      {"epochs": sorted({r["_epoch"] for r in batch}), "rng": draw})
        time.sleep(a.sleep)                           # rollouts + optimizer step
        stream.state.step = step + 1
        save_state(state_path, stream.state, {"sampling_rng": rng.bit_generator.state})
    print("done", flush=True)


if __name__ == "__main__":
    main()
