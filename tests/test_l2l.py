"""L2L engine vs plain HF transformers Qwen3 on a tiny random model (CPU)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def res():
    from soupgrpo.equivalence import run_all
    return run_all(n_layers=3, gen_prompts=20, verbose=False)


def test_fp32_logits_and_grads(res):
    assert res["fp32"]["logits_nmax"] <= 1e-5
    assert res["fp32"]["grad_nmax_max"] <= 1e-5
    assert abs(res["fp32"]["loss_engine"] - res["fp32"]["loss_ref"]) <= 1e-4 * abs(res["fp32"]["loss_ref"])


def test_bf16_within_noise_floor(res):
    for m in ("nmax", "rfro"):
        b = res["bf16"][m]
        assert b["l2l_logits"] <= 1.5 * b["noise_logits"]
        assert b["l2l_grads"] <= 1.5 * b["noise_grads"]


def test_kv_greedy_identical(res):
    assert res["generation_greedy"]["PASS"], res["generation_greedy"]["first_diff"]


def test_copies_once_per_layer_per_pass(res):
    assert res["copies"]["PASS"], res["copies"]


def test_padding_invariance(res):
    assert res["padding_invariance_nmax"] <= 1e-5


def test_sampling_logp_matches_training_forward(res):
    assert res["gen_vs_train_logp_nmax"] <= 1e-5


def test_reference_is_adapter_off(res):
    assert res["ref_logp_vs_adapter_off"] <= 1e-6


def test_buffer_variants_identical(res):
    assert max(res["variants_nmax"].values()) <= 1e-5


def test_fp16_mode_at_least_as_accurate_as_bf16():
    from soupgrpo.equivalence import fp16_mode_check
    r = fp16_mode_check(n_layers=3)
    assert r["residual_dtype"] == "torch.float32"
    assert r["nonfinite_events"] == 0
    assert r["PASS"], r


def test_boundary_store_reserved_once(tmp_path):
    """E14: the per-layer input buffer must be reserved once and never grow
    during training (growing it ran the Kaggle host out of RAM)."""
    import torch
    from soupgrpo.grpo import make_loss_fn
    from soupgrpo.l2l import L2LEngine
    from soupgrpo.lora import LoRA
    from soupgrpo.tiny import make_tiny_qwen3
    from soupgrpo.weights import WeightStore
    st = WeightStore(make_tiny_qwen3(tmp_path / "m", n_layers=2), dtype=torch.float32)
    eng = L2LEngine(st, lora=LoRA(st.dims, r=4, alpha=8, seed=1, b_init_std=0.05))
    d = st.dims
    eng.boundaries.reserve(d.n_layers, 4 * 12 * d.hidden, eng.res_dtype)
    for T in (7, 12, 9):                      # different lengths, all within the reserved size
        ids = torch.randint(3, 500, (4, T))
        attn = torch.ones(4, T, dtype=torch.long)
        lm = torch.zeros(4, T, dtype=torch.bool)
        lm[:, T // 2:] = True
        eng.train_pass(ids, attn, lm, make_loss_fn(torch.randn(4), 0.0, 1.0, {}), micro_batch=2)
    assert eng.boundaries.allocations == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in eng.lora.parameters())
