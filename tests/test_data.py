import json
from pathlib import Path

from soupgrpo.countdown import puzzle_key, read_jsonl
from soupgrpo.reward import countdown_reward

DATA = Path(__file__).resolve().parents[1] / "data"


def _load(prefix):
    return {f.name: read_jsonl(f) for f in DATA.glob(f"{prefix}_*.jsonl")}


def _training_files():
    """Files training reads: train_<lvl>_main.jsonl replaces train_<lvl>.jsonl
    when it exists (its last 100 rows became pilot_<lvl>_extra, split_extra.py)."""
    t = _load("train")
    for name in [n for n in t if n.endswith("_main.jsonl")]:
        t.pop(name.replace("_main", ""), None)
    return t


def test_no_train_eval_overlap():
    train = {p.key() for ps in _training_files().values() for p in ps}
    evals = {p.key() for pre in ("heldout", "pilot") for ps in _load(pre).values() for p in ps}
    assert train and evals
    assert not (train & evals)


def test_reference_solutions_valid_and_unique():
    for name, ps in {**_load("train"), **_load("heldout"), **_load("pilot")}.items():
        keys = [p.key() for p in ps]
        assert len(keys) == len(set(keys)), f"duplicates in {name}"
        for p in ps:
            assert countdown_reward(f"<answer>{p.solution}</answer>", p.numbers, p.target).reward == 1.0


def test_manifest_matches_files():
    import hashlib
    m = json.loads((DATA / "MANIFEST.json").read_text())
    for fname, info in m["files"].items():
        assert hashlib.sha256((DATA / fname).read_bytes()).hexdigest() == info["sha256"], fname


def test_key_ignores_order():
    assert puzzle_key([1, 2, 3], 6) == puzzle_key([3, 1, 2], 6)
    assert puzzle_key([1, 2, 3], 6) != puzzle_key([1, 2, 3], 7)
