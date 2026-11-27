"""DFlash speculative decoding on a tiny random Qwen3 (CPU, fp32).

Correctness does not depend on the drafter being good: a random drafter
(almost nothing accepted), an oracle drafter (everything accepted) and a
half-oracle (partial acceptance) must all give exactly the plain greedy output.
Sampling: the acceptance rule must leave the target's distribution unchanged.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from soupgrpo.dflash import Drafter
    from soupgrpo.l2l import L2LEngine
    from soupgrpo.lora import LoRA
    from soupgrpo.tiny import make_tiny_qwen3
    from soupgrpo.weights import WeightStore
    st = WeightStore(make_tiny_qwen3(tmp_path_factory.mktemp("m"), n_layers=6), dtype=torch.float32)
    eng = L2LEngine(st, lora=LoRA(st.dims, r=4, alpha=8, seed=1, b_init_std=0.05))
    dr = Drafter.random(st.dims, n_layers=2, target_layer_ids=[1, 3], block_size=5, mask_token_id=7)
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(3, 500, (int(n),), generator=g).tolist() for n in torch.randint(3, 12, (10,), generator=g)]
    return eng, dr, prompts


def _plain(eng, prompts, n, **kw):
    return eng.generate(prompts, n, 0.0, stop_ids=kw.get("stops", ()), pad_id=0, batch_size=len(prompts))


def test_greedy_identical_random_drafter(setup):
    from soupgrpo.dflash import DFlashGenerator
    eng, dr, prompts = setup
    ref = _plain(eng, prompts, 20)
    g = DFlashGenerator(eng, dr)
    out = g.generate(prompts, 20, 0.0, stop_ids=(), pad_id=0, batch_size=4)
    assert [o.tokens for o in out] == [o.tokens for o in ref]
    for o, r in zip(out, ref):
        assert max(abs(a - b) for a, b in zip(o.logprobs, r.logprobs)) < 1e-4


def _oracle(ref_tokens, corrupt_at=None):
    def fn(out, n, k):
        B = out.shape[0]
        d = torch.zeros((B, k), dtype=torch.long)
        for b in range(B):
            for j in range(k):
                i = int(n[b]) + j
                d[b, j] = ref_tokens[b][i] if i < len(ref_tokens[b]) else 0
                if corrupt_at is not None and j == corrupt_at:
                    d[b, j] = (d[b, j] + 1) % 500
        return d, None
    return fn


@pytest.mark.parametrize("corrupt", [None, 2])
def test_greedy_identical_oracle_drafter(setup, corrupt):
    from soupgrpo.dflash import DFlashGenerator
    eng, dr, prompts = setup
    ref = [o.tokens for o in _plain(eng, prompts, 23)]
    g = DFlashGenerator(eng, dr)
    out = g.generate(prompts, 23, 0.0, stop_ids=(), pad_id=0, batch_size=len(prompts), draft_fn=_oracle(ref, corrupt))
    assert [o.tokens for o in out] == ref
    expect = 5 if corrupt is None else 3          # all 4 drafts + bonus, or 2 drafts + bonus
    full = [a for a in g.stats.accept_lens]
    assert max(full) == expect and full.count(expect) >= len(full) - len(prompts)  # last block may be cut by max_new


def test_stop_tokens_and_lora_off(setup):
    from soupgrpo.dflash import DFlashGenerator
    eng, dr, prompts = setup
    plain = _plain(eng, prompts, 30)
    stops = tuple({o.tokens[5] for o in plain[:3]})      # tokens that really occur -> some rows stop early
    ref = eng.generate(prompts, 30, 0.0, stop_ids=stops, pad_id=0, batch_size=len(prompts), lora_enabled=False)
    g = DFlashGenerator(eng, dr)
    out = g.generate(prompts, 30, 0.0, stop_ids=stops, pad_id=0, batch_size=3, lora_enabled=False,
                     draft_fn=_oracle([o.tokens for o in ref]))
    assert [o.tokens for o in out] == [o.tokens for o in ref]
    assert [o.finish for o in out] == [o.finish for o in ref]
    assert any(o.finish == "eos" for o in ref)


def test_speculative_accept_is_exact():
    """Synthetic p, q over 6 tokens, 2 draft positions: the committed tokens must
    follow p exactly (first token from p[0]; second, given the first was
    accepted, from p[1])."""
    from soupgrpo.dflash import speculative_accept
    g = torch.Generator().manual_seed(0)
    N, V, k = 400_000, 6, 2
    p = torch.softmax(torch.randn(1, k + 1, V, generator=g) * 1.5, -1).expand(N, -1, -1)
    q = torch.softmax(torch.randn(1, k, V, generator=g) * 1.5, -1).expand(N, -1, -1)
    draft = torch.multinomial(q.reshape(-1, V), 1, generator=g).view(N, k)
    u = torch.rand((N, k), generator=g)
    n, nxt = speculative_accept(p, q, draft, u)
    tok_next = torch.multinomial(nxt, 1, generator=g)[:, 0]
    first = torch.where(n >= 1, draft[:, 0], tok_next)
    freq = torch.bincount(first, minlength=V).float() / N
    assert (freq - p[0, 0]).abs().max() < 0.004, (freq, p[0, 0])
    # second committed token, among rows whose first draft was accepted
    m = n >= 1
    second = torch.where(n[m] >= 2, draft[m, 1], tok_next[m])
    f2 = torch.bincount(second, minlength=V).float() / int(m.sum())
    assert (f2 - p[0, 1]).abs().max() < 0.006, (f2, p[0, 1])


def test_sampling_matches_plain_distribution(setup):
    """End to end at T = 0.7: token frequencies at positions 1-5 (plain sampling
    vs DFlash with the random drafter) agree; the deliberately biased
    'accept_all' rule must be detected by the same test (power check)."""
    from soupgrpo.dflash import DFlashGenerator
    from soupgrpo.stats import chi2_homogeneity
    eng, dr, prompts = setup
    P = [prompts[0]] * 3000
    plain = eng.generate(P, 6, 0.7, (), 0, 3000, torch.Generator().manual_seed(1))
    spec = DFlashGenerator(eng, dr).generate(P, 6, 0.7, (), 0, 3000, torch.Generator().manual_seed(2))
    bad = DFlashGenerator(eng, dr, mode="accept_all").generate(P, 6, 0.7, (), 0, 3000, torch.Generator().manual_seed(3))
    for pos in range(1, 6):
        a = [o.tokens[pos] for o in plain]
        assert chi2_homogeneity(a, [o.tokens[pos] for o in spec])["p"] > 0.001
    assert min(chi2_homogeneity([o.tokens[pos] for o in plain], [o.tokens[pos] for o in bad])["p"]
               for pos in range(1, 6)) < 1e-6


def test_loader_roundtrip(setup, tmp_path):
    """The checkpoint-format loader gives the same drafter as the one written."""
    from soupgrpo.dflash import DFlashGenerator, Drafter, write_random_drafter
    eng, _, prompts = setup
    d = write_random_drafter(tmp_path / "dr", eng.store.model_dir, 2, (1, 3), 5, 7, seed=3)
    a = Drafter.load(d, "cpu", torch.float32)
    b = Drafter.random(eng.d, 2, [1, 3], 5, 7, seed=3)
    assert a.target_layer_ids == [1, 3] and a.block_size == 5 and a.mask_token_id == 7
    for k in b.layers[1]:
        assert torch.equal(a.layers[1][k], b.layers[1][k]), k
    oa = DFlashGenerator(eng, a).generate(prompts[:3], 8, 0.7, (), 0, 3, torch.Generator().manual_seed(5))
    ob = DFlashGenerator(eng, b).generate(prompts[:3], 8, 0.7, (), 0, 3, torch.Generator().manual_seed(5))
    assert [o.tokens for o in oa] == [o.tokens for o in ob]


def test_unwritten_kv_slots_cannot_leak(setup, monkeypatch):
    """Regression (local notebook test): rows that accept fewer tokens than the
    longest row see never-written KV slots in the verify pass. If those hold NaN,
    masked attention still turns NaN (additive -inf mask). Simulate the worst
    allocator: every torch.empty is NaN-filled."""
    from soupgrpo.dflash import DFlashGenerator
    eng, dr, prompts = setup
    real_empty = torch.empty

    def nan_empty(*a, **k):
        t = real_empty(*a, **k)
        return t.fill_(float("nan")) if t.is_floating_point() else t
    ref = [o.tokens for o in _plain(eng, prompts, 20)]
    monkeypatch.setattr(torch, "empty", nan_empty)
    full, part = _oracle(ref), _oracle(ref, 0)

    def mixed(out, n, k):          # even rows accept everything, odd rows nothing -> rows drift apart
        a, _ = full(out, n, k)
        b, _ = part(out, n, k)
        a[1::2] = b[1::2]
        return a, None
    g = DFlashGenerator(eng, dr)
    out = g.generate(prompts, 20, 0.0, (), 0, len(prompts), draft_fn=mixed)
    assert [o.tokens for o in out] == ref
    assert min(g.stats.accept_lens) == 1 and max(g.stats.accept_lens) == 5


@pytest.mark.parametrize("which", ["plain", "dflash"])
def test_oom_split_frees_failed_attempt(setup, monkeypatch, which):
    """Regression E19 (Kaggle sweep, batch 48): when a batch runs out of memory,
    the half-size retries must not run while the failed attempt's tensors (its
    KV cache) are still alive. Fake an OOM after the 'KV cache' is allocated and
    check, at the start of each retry, that the failed attempts' caches are gone."""
    import gc
    import weakref
    from soupgrpo.dflash import DFlashGenerator
    eng, dr, prompts = setup
    obj = eng if which == "plain" else DFlashGenerator(eng, dr)
    name = "_generate_batch" if which == "plain" else "_batch"
    real = getattr(obj, name)
    failed = []

    def fake(ps, *a):
        gc.collect()
        alive = [r for r in failed if r() is not None]
        assert not alive, f"{len(alive)} failed attempt(s) still hold their KV cache during a retry"
        if len(ps) > 2:
            kv = torch.zeros(1000)          # stands in for the failed batch's KV cache
            failed.append(weakref.ref(kv))
            raise torch.OutOfMemoryError("fake")
        return real(ps, *a)
    monkeypatch.setattr(obj, name, fake)
    out = obj.generate(prompts[:8], 4, 0.0, stop_ids=(), pad_id=0, batch_size=8)
    assert len(out) == 8 and obj.oom_splits == 3
    assert len(failed) == 3                  # 8 -> 4+4 -> each 4 -> 2+2
