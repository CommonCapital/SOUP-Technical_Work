import json

from soupgrpo.data_stream import (PromptStream, StreamState, append_id_log, load_state,
                                  read_id_log, reconcile_id_log, save_state)


def _data(tmp_path, n=10):
    p = tmp_path / "d.jsonl"
    p.write_text("".join(json.dumps({"id": f"r{i}"}) + "\n" for i in range(n)))
    return p


def test_epoch_is_a_permutation(tmp_path):
    s = PromptStream(_data(tmp_path), seed=1)
    ids = [r["id"] for r in s.next_batch(10)]
    assert sorted(ids) == sorted(f"r{i}" for i in range(10))
    assert s.state.epoch == 1 and s.state.position == 0


def test_batch_crosses_epoch(tmp_path):
    s = PromptStream(_data(tmp_path), seed=1)
    s.next_batch(7)
    b = s.next_batch(7)
    assert [r["_epoch"] for r in b] == [0, 0, 0, 1, 1, 1, 1]


def test_epochs_use_different_orders(tmp_path):
    s = PromptStream(_data(tmp_path, 50), seed=1)
    e0 = [r["id"] for r in s.next_batch(50)]
    e1 = [r["id"] for r in s.next_batch(50)]
    assert e0 != e1 and sorted(e0) == sorted(e1)


def test_resume_from_saved_state_continues_exactly(tmp_path):
    d = _data(tmp_path, 23)
    ref = PromptStream(d, seed=5)
    ref_ids = [r["id"] for _ in range(9) for r in ref.next_batch(4)]

    a = PromptStream(d, seed=5)
    first = [r["id"] for _ in range(4) for r in a.next_batch(4)]
    a.state.step = 4
    save_state(tmp_path / "st.json", a.state, {"x": 1})
    st, extra = load_state(tmp_path / "st.json")
    b = PromptStream(d, seed=5, state=st)
    rest = [r["id"] for _ in range(5) for r in b.next_batch(4)]
    assert first + rest == ref_ids and extra == {"x": 1}


def test_state_rejects_changed_file(tmp_path):
    d = _data(tmp_path, 10)
    st = StreamState(seed=1, n_records=11)
    try:
        PromptStream(d, seed=1, state=st)
        raise AssertionError("should fail")
    except ValueError:
        pass


def test_reconcile_drops_uncommitted_and_torn_lines(tmp_path):
    log = tmp_path / "ids.jsonl"
    for s in range(5):
        append_id_log(log, s, [f"a{s}"])
    with open(log, "a") as f:
        f.write('{"step": 5, "ids": ["a')  # torn write from kill -9
    dropped = reconcile_id_log(log, committed_steps=3)
    assert dropped == 3
    assert [s for s, _ in read_id_log(log)] == [0, 1, 2]
