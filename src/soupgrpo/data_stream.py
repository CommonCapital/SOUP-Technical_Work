"""Streaming prompt loader with exact (deterministic) resume.

Prompts stay on disk. At start we build a byte-offset index of the JSONL file
(one pass, no records kept in memory) and read each record by seeking to its
offset when it is needed.

Order: epoch e uses permutation  rng(seed, e).permutation(N).  The permutation
is a pure function of (seed, epoch), so the only state that must be saved is
(epoch, position). A batch that crosses the end of an epoch takes the rest of
epoch e and continues at the start of epoch e+1.

Resume contract (plan step 5, hypothesis H19):
  - state is saved atomically (tmp file + fsync + os.replace) AFTER the
    optimizer step of that step finishes, together with the step number;
  - the sample-ID log is append-only and every line carries its step number;
    on resume, lines for steps after the last saved step are dropped (those
    steps were attempted but never committed, and will run again);
  - proof = reconciled ID log of a killed+resumed run == log of an
    uninterrupted run (no duplicates, no gaps).
The RNG used for sampling rollouts is saved in the same state file by the
trainer (extra_state), so generation after resume is reproducible too.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StreamState:
    seed: int
    epoch: int = 0
    position: int = 0      # index into the epoch's permutation
    step: int = 0          # number of committed (finished) steps
    n_records: int = 0     # sanity check on resume: same file

    def to_dict(self) -> dict:
        return dict(seed=self.seed, epoch=self.epoch, position=self.position,
                    step=self.step, n_records=self.n_records)


class PromptStream:
    def __init__(self, path: str | os.PathLike, seed: int, state: StreamState | None = None):
        self.path = Path(path)
        self.offsets = self._index(self.path)
        n = len(self.offsets)
        self.state = state or StreamState(seed=seed, n_records=n)
        if self.state.n_records != n:
            raise ValueError(f"data file changed: state has {self.state.n_records} records, file has {n}")
        if self.state.seed != seed:
            raise ValueError(f"seed mismatch: state {self.state.seed} vs {seed}")
        self._perm_epoch = -1
        self._perm = None
        self._fh = open(self.path, "rb")

    @staticmethod
    def _index(path: Path) -> np.ndarray:
        offsets = []
        pos = 0
        with open(path, "rb") as f:
            for line in f:
                if line.strip():
                    offsets.append(pos)
                pos += len(line)
        return np.asarray(offsets, dtype=np.int64)

    def _perm_for(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:
            rng = np.random.default_rng([self.state.seed, epoch])
            self._perm = rng.permutation(len(self.offsets))
            self._perm_epoch = epoch
        return self._perm

    def _read(self, idx: int) -> dict:
        self._fh.seek(int(self.offsets[idx]))
        return json.loads(self._fh.readline())

    def next_batch(self, batch_size: int) -> list[dict]:
        """Return the next batch and advance the in-memory cursor.
        The cursor is only made durable by save()."""
        out = []
        n = len(self.offsets)
        while len(out) < batch_size:
            perm = self._perm_for(self.state.epoch)
            take = min(batch_size - len(out), n - self.state.position)
            for k in perm[self.state.position:self.state.position + take]:
                rec = self._read(int(k))
                rec["_epoch"] = self.state.epoch
                out.append(rec)
            self.state.position += take
            if self.state.position == n:
                self.state.epoch += 1
                self.state.position = 0
        return out

    def close(self) -> None:
        self._fh.close()


# ---------- durable state ----------

def atomic_write_json(path: str | os.PathLike, obj: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # atomic on POSIX
    # fsync the directory so the rename itself survives a crash
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def save_state(path, stream_state: StreamState, extra_state: dict | None = None) -> None:
    atomic_write_json(path, {"stream": stream_state.to_dict(), "extra": extra_state or {}})


def load_state(path) -> tuple[StreamState, dict] | None:
    path = Path(path)
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return StreamState(**d["stream"]), d.get("extra", {})


# ---------- sample-ID log ----------

def append_id_log(path, step: int, ids: list[str], extra: dict | None = None) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"step": step, "ids": ids, **(extra or {})}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def reconcile_id_log(path, committed_steps: int) -> int:
    """Drop log lines for steps >= committed_steps (attempted, not committed)
    and any torn last line. Returns the number of lines dropped."""
    path = Path(path)
    if not path.exists():
        return 0
    keep, dropped = [], 0
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if rec["step"] < committed_steps:
            keep.append(line)
        else:
            dropped += 1
    atomic_write_text(path, "".join(l + "\n" for l in keep))
    return dropped


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_id_log(path) -> list[tuple[int, list[str]]]:
    out = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        out.append((rec["step"], rec["ids"]))
    return out
