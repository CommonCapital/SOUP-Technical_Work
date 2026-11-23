"""Generate all Countdown splits and check them.

For every level: train (1,500), heldout (500), pilot (100), each from its own
seed. Plus a 'harder' heldout split one number above each level, used for the
generalisation check. Asserts:
  - no puzzle key appears in both a training set and any evaluation set
  - every stored reference solution scores 1.0 with the reward function
Writes data/<split>_<level>.jsonl and data/MANIFEST.json (sha256 per file).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soupgrpo.countdown import LEVELS, generate_split, write_jsonl  # noqa: E402
from soupgrpo.reward import countdown_reward  # noqa: E402

SIZES = {"train": 1500, "heldout": 500, "pilot": 100}
HARDER_OF = {"n3": "n4", "n4": "n5", "n5": "n6"}


def main() -> None:
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    splits: dict[str, list] = {}

    # Evaluation-type splits first, then train excludes all of their keys.
    for lvl in LEVELS.values():
        splits[f"heldout_{lvl.name}"] = generate_split(lvl, SIZES["heldout"], "heldout")
        splits[f"pilot_{lvl.name}"] = generate_split(lvl, SIZES["pilot"], "pilot")
    eval_keys = {p.key() for name, ps in splits.items() for p in ps}
    for lvl in LEVELS.values():
        splits[f"train_{lvl.name}"] = generate_split(lvl, SIZES["train"], "train", exclude_keys=eval_keys)

    # Checks
    train_keys = {p.key() for n, ps in splits.items() if n.startswith("train_") for p in ps}
    overlap = train_keys & eval_keys
    assert not overlap, f"train/eval overlap: {len(overlap)}"
    bad = 0
    for ps in splits.values():
        for p in ps:
            r = countdown_reward(f"<answer>{p.solution}</answer>", p.numbers, p.target)
            bad += r.reward != 1.0
    assert bad == 0, f"{bad} reference solutions fail the reward"

    manifest = {"harder_split_of": HARDER_OF, "files": {}}
    for name, ps in sorted(splits.items()):
        path = out / f"{name}.jsonl"
        write_jsonl(path, ps)
        manifest["files"][path.name] = {
            "n": len(ps),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(splits)} files; train/eval overlap = 0; all {sum(map(len, splits.values()))} reference solutions score 1.0")
    for name in sorted(splits):
        ps = splits[name]
        ts = [p.target for p in ps]
        print(f"  {name:22s} n={len(ps):5d} target min/median/max = {min(ts)}/{sorted(ts)[len(ts)//2]}/{max(ts)}")


if __name__ == "__main__":
    main()
