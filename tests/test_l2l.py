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
