"""Unit tests for ``dreamer_arm.core.distributions``.

We pin the two non-trivial invariants from the Dreamer codebase: ``symlog``
and ``symexp`` are mutual inverses, and the ``TwoHot`` distribution
round-trips known scalar means.
"""

from __future__ import annotations

import torch

from dreamer_arm.core.distributions import TanhNormal, bounded_normal, symexp_twohot
from dreamer_arm.utils.tensor import symexp, symlog


def test_symlog_symexp_inverse_on_grid() -> None:
    x = torch.linspace(-10.0, 10.0, 41)
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-5)
    assert torch.allclose(symlog(symexp(x)), x, atol=1e-5)


def test_twohot_uniform_logits_mode_is_zero() -> None:
    """A uniform two-hot over a symmetric symexp grid has mode 0."""
    bin_num = 21  # odd so 0.0 is a bin centre
    logits = torch.zeros(1, bin_num)
    dist = symexp_twohot(logits, bin_num=bin_num)
    assert torch.allclose(dist.mode(), torch.zeros(1, 1), atol=1e-4)


def test_bounded_normal_is_a_true_tanh_distribution() -> None:
    """Samples must not create hard-clipped atoms at the action bounds."""
    torch.manual_seed(0)
    logits = torch.zeros(100_000, 4 * 2)
    dist = bounded_normal(logits, min_std=0.1, max_std=1.0)

    assert isinstance(dist, TanhNormal)
    sample = dist.rsample()
    assert (sample > -1.0).all()
    assert (sample < 1.0).all()
    assert not (sample.abs() >= 1.0 - 1e-6).any()
    assert torch.isfinite(dist.log_prob(sample)).all()


def test_bounded_normal_mode_preserves_intended_tanh_mean() -> None:
    raw_mean = torch.tensor([[0.5, -1.0]])
    raw_std = torch.zeros_like(raw_mean)
    dist = bounded_normal(torch.cat([raw_mean, raw_std], dim=-1), min_std=0.1, max_std=1.0)

    assert torch.allclose(dist.mode, torch.tanh(raw_mean))
    assert torch.equal(dist.pre_mean, raw_mean)


def test_tanh_normal_log_prob_includes_change_of_variables() -> None:
    mean = torch.tensor([[0.2, -0.4]])
    std = torch.tensor([[0.5, 0.7]])
    value = torch.tensor([[0.1, -0.8]])
    ours = TanhNormal(mean, std).log_prob(value)
    reference = torch.distributions.Independent(
        torch.distributions.TransformedDistribution(
            torch.distributions.Normal(mean, std),
            [torch.distributions.TanhTransform(cache_size=1)],
        ),
        1,
    ).log_prob(value)
    assert torch.allclose(ours, reference, atol=1e-5)
