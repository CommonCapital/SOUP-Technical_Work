"""GRPO math: group advantages, sequence packing, loss.

Advantage (per prompt group of G answers): A_i = (r_i - mean) / (std + eps),
or (r_i - mean) if scale_by_std=False. If all answers in a group score the
same, every A_i = 0 and the group teaches nothing ("zero-advantage group").

Loss, one optimizer update per batch (so the PPO ratio is exactly 1 and
clipping never activates; old-policy log-probs are free):
  L = sum_i sum_t [ -A_i * exp(logp_it - stopgrad(logp_it)) + beta * KL_it ] / Z
  KL_it = exp(ref - logp) - (ref - logp) - 1      (k3 estimator, >= 0)
  Z = n_sequences * max_new_tokens                (constant: no length bias)
The exp(logp - stopgrad(logp)) term equals 1 but has gradient d logp, which is
the policy-gradient term.

Groups with zero advantage are dropped from the training pass when beta = 0
(their gradient is exactly zero), which saves compute.
"""

from __future__ import annotations

import numpy as np
import torch


def group_advantages(rewards: np.ndarray, group: int, scale_by_std: bool = True, eps: float = 1e-4) -> np.ndarray:
    r = rewards.reshape(-1, group).astype(np.float64)
    mean = r.mean(1, keepdims=True)
    adv = r - mean
    if scale_by_std:
        adv = adv / (r.std(1, keepdims=True) + eps)
    return adv.reshape(-1)


def shuffled_rewards(rewards: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Negative control: same reward values, randomly reassigned across the
    whole batch, so they carry no information about the answer."""
    return rewards[rng.permutation(len(rewards))]


def pack_sequences(prompts: list[list[int]], completions: list[list[int]], pad_id: int):
    """Right-padded [N, T] ids / attention / loss mask (loss on completion tokens).
    Positions come from the attention mask, so the padding side is irrelevant
    for the math (tested in the equivalence suite)."""
    seqs = [p + c for p, c in zip(prompts, completions)]
    T = max(len(s) for s in seqs)
    N = len(seqs)
    ids = torch.full((N, T), pad_id, dtype=torch.long)
    attn = torch.zeros((N, T), dtype=torch.long)
    lm = torch.zeros((N, T), dtype=torch.long)
    for i, (p, s) in enumerate(zip(prompts, seqs)):
        ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        lm[i, len(p):len(s)] = 1
    return ids, attn, lm


def make_loss_fn(adv: torch.Tensor, beta: float, norm: float, stats: dict):
    """adv: [N] advantages on the engine device. Returns loss_fn for
    L2LEngine.train_pass and accumulates KL statistics into `stats`."""

    def loss_fn(lp, ref_lp, b, t):
        a = adv[b]
        pg = -(a * torch.exp(lp - lp.detach()))
        loss = pg
        if beta > 0 and ref_lp is not None:
            d = ref_lp - lp
            kl = torch.exp(d) - d - 1
            loss = loss + beta * kl
            stats["kl_sum"] = stats.get("kl_sum", 0.0) + float(kl.detach().sum())
        elif ref_lp is not None:
            d = (ref_lp - lp).detach()
            stats["kl_sum"] = stats.get("kl_sum", 0.0) + float((torch.exp(d) - d - 1).sum())
        stats["n_tok"] = stats.get("n_tok", 0) + len(b)
        return loss.sum() / norm

    return loss_fn
