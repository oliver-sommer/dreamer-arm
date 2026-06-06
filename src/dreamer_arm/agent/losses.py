"""Pure-function loss components used by the Dreamer agent.

Keeping these here (rather than as methods on ``Dreamer``) makes them
trivial to unit-test on synthetic inputs without spinning up the full
world model.
"""

from __future__ import annotations

import torch

from dreamer_arm.utils.tensor import to_f32


def barlow_twins_loss(
    z1: torch.Tensor, z2: torch.Tensor, lambd: float, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Barlow-Twins redundancy-reduction loss between two ``(N, D)`` embeddings.

    Returns ``(total, invariance, redundancy)`` so that callers can log the
    two components separately. The total is
    ``Σ_i (1 - C_ii)^2 + lambd · Σ_{i≠j} C_ij^2`` where ``C`` is the
    cross-correlation matrix of standardised features (cf. Zbontar et al.,
    2021; eq. 5 of the R2-Dreamer paper).
    """
    assert z1.shape == z2.shape, (z1.shape, z2.shape)
    n, d = z1.shape
    z1_n = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + eps)
    z2_n = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + eps)
    c = (z1_n.T @ z2_n) / n
    invariance = (torch.diagonal(c) - 1.0).pow(2).sum()
    off_diag_mask = ~torch.eye(d, dtype=torch.bool, device=z1.device)
    redundancy = c[off_diag_mask].pow(2).sum()
    return invariance + lambd * redundancy, invariance, redundancy


def lambda_return(
    last: torch.Tensor,
    term: torch.Tensor,
    reward: torch.Tensor,
    value: torch.Tensor,
    boot: torch.Tensor,
    disc: float,
    lamb: float,
) -> torch.Tensor:
    """λ-return over a length-T trajectory in ``(B, T, 1)`` form.

    ``lamb=1`` → discounted Monte-Carlo return; ``lamb=0`` → 1-step return.
    ``term`` and ``last`` are episode-end indicator tensors (terminal vs
    truncation respectively); ``boot`` is the bootstrap value series.
    """
    assert last.shape == term.shape == reward.shape == value.shape == boot.shape, (
        last.shape,
        term.shape,
        reward.shape,
        value.shape,
        boot.shape,
    )
    live = (1.0 - to_f32(term))[:, 1:] * disc
    cont = (1.0 - to_f32(last))[:, 1:] * lamb
    interm = reward[:, 1:] + (1.0 - cont) * live * boot[:, 1:]
    out = [boot[:, -1]]
    for i in reversed(range(live.shape[1])):
        out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
    return torch.stack(list(reversed(out))[:-1], dim=1)
