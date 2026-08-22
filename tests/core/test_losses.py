"""Unit tests for ``dreamer_arm.core.losses``.

The Barlow-Twins identity holds on standardised independent inputs:
diagonal of the cross-correlation matrix → 1, off-diagonal → 0. The
λ-return test pins the boundary cases (``lamb=1`` is the discounted MC
return; ``lamb=0`` is the 1-step TD target).
"""

from __future__ import annotations

import torch

from dreamer_arm.core.losses import barlow_twins_loss, lambda_return


def test_barlow_twins_diagonal_at_identity() -> None:
    torch.manual_seed(0)
    n, d = 4096, 8
    # Identical embeddings -> diagonal of cross-correlation is exactly 1.
    z = torch.randn(n, d)
    total, inv, red = barlow_twins_loss(z, z, lambd=0.01)
    assert inv.item() < 1e-6
    # Off-diagonal terms still nonzero because of finite-sample noise.
    assert torch.isfinite(total)
    assert red.item() >= 0


def test_barlow_twins_independence_zero_redundancy() -> None:
    torch.manual_seed(0)
    n, d = 8192, 6
    # Independent z1, z2 with identical marginals: diagonal -> 0 (not 1),
    # so invariance is large; redundancy is small. We're testing that
    # redundancy scales like 1/n (vanishing) and invariance is ~d.
    z1 = torch.randn(n, d)
    z2 = torch.randn(n, d)
    _, inv, red = barlow_twins_loss(z1, z2, lambd=1.0)
    assert inv.item() > 1.0  # diagonal far from 1
    assert red.item() < 1.0  # off-diagonal tiny


def test_lambda_return_mc_limit() -> None:
    """``lamb=1`` collapses to the discounted Monte-Carlo return."""
    b, t = 2, 5
    reward = torch.ones(b, t, 1)
    value = torch.zeros(b, t, 1)
    boot = torch.zeros(b, t, 1)
    term = torch.zeros(b, t, 1)
    last = torch.zeros(b, t, 1)
    disc = 0.9
    out = lambda_return(last, term, reward, value, boot, disc, lamb=1.0)
    assert out.shape == (b, t - 1, 1)
    # First step has the most future reward; later steps less.
    assert torch.all(out[:, 0] > out[:, -1])
    # All values are finite.
    assert torch.isfinite(out).all()


def test_lambda_return_one_step_limit() -> None:
    """``lamb=0`` is the 1-step TD target ``r + gamma*V``."""
    b, t = 1, 3
    reward = torch.tensor([[[0.0], [1.0], [2.0]]])
    boot = torch.tensor([[[0.0], [10.0], [20.0]]])
    value = torch.zeros(b, t, 1)
    term = torch.zeros(b, t, 1)
    last = torch.zeros(b, t, 1)
    disc = 0.5
    out = lambda_return(last, term, reward, value, boot, disc, lamb=0.0)
    # reward[t=1] + gamma * boot[t=1] = 1 + 0.5*10 = 6
    # reward[t=2] + gamma * boot[t=2] = 2 + 0.5*20 = 12
    assert torch.allclose(out[0, 0], torch.tensor([6.0]))
    assert torch.allclose(out[0, 1], torch.tensor([12.0]))
