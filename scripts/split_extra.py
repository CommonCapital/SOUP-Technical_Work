"""Pilot rule 4 (PREDICTIONS.md, "Pilot 2"): the chosen level was within 10
points of an edge, so 100 more pilot prompts are taken from its train split and
removed from training. Deterministic: the LAST 100 rows (the generator is
random, so they are a fair sample). Updates data/MANIFEST.json.

  python scripts/split_extra.py n5
writes data/pilot_<lvl>_extra.jsonl (100) and data/train_<lvl>_main.jsonl (rest).
"""
import hashlib
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data"
lvl = sys.argv[1]
rows = (D / f"train_{lvl}.jsonl").read_text().splitlines()
parts = {f"train_{lvl}_main.jsonl": rows[:-100], f"pilot_{lvl}_extra.jsonl": rows[-100:]}
man = json.loads((D / "MANIFEST.json").read_text())
for name, lines in parts.items():
    text = "\n".join(lines) + "\n"
    (D / name).write_text(text)
    man["files"][name] = {"n": len(lines), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "derived_from": f"train_{lvl}.jsonl"}
ids_main = {json.loads(l)["id"] for l in parts[f"train_{lvl}_main.jsonl"]}
ids_extra = {json.loads(l)["id"] for l in parts[f"pilot_{lvl}_extra.jsonl"]}
assert not ids_main & ids_extra
(D / "MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
print({k: len(v) for k, v in parts.items()}, "overlap 0")
