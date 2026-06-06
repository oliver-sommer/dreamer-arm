"""Adaptive Gradient Clipping (Brock et al., 2021).

Scales each parameter's gradient by ``min(1, clip * max(||p||_2, pmin) / ||g||_2)``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def adaptive_grad_clip(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    clip: float,
    pmin: float,
) -> None:
    """In-place AGC over ``parameters``. Skips params with no gradient."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad
        pnorm = torch.norm(p.detach(), p=2)
        gnorm = torch.norm(g, p=2)
        upper = clip * torch.maximum(torch.tensor(pmin, device=pnorm.device, dtype=pnorm.dtype), pnorm)
        scale = 1.0 / torch.maximum(torch.tensor(1.0, device=gnorm.device, dtype=gnorm.dtype), gnorm / upper)
        g.detach().mul_(scale)
