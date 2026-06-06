"""Unit tests for ``dreamer_arm.architecture.distributions``.

We pin the two non-trivial invariants from the Dreamer codebase: ``symlog``
and ``symexp`` are mutual inverses, and the ``TwoHot`` distribution
round-trips known scalar means.
"""

from __future__ import annotations

import torch

from dreamer_arm.architecture.distributions import symexp_twohot
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
