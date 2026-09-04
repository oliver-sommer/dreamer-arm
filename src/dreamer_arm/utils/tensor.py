"""Tensor primitives shared across the agent: symlog / symexp, dtype helpers, padding."""

from __future__ import annotations

import numpy as np
import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Sign-preserving log: sign(x) * log(1 + |x|). Stable for large |x|."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float32)


def to_i32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.int32)


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def rpad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Right-pad a tensor with ``pad`` singleton trailing axes."""
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x


def compute_global_norm(tensors: list[torch.Tensor | None]) -> torch.Tensor:
    present = [t for t in tensors if t is not None]
    if not present:
        return torch.tensor(0.0)
    square_sum = torch.stack([t.detach().float().square().sum() for t in present]).sum()
    return square_sum.sqrt()


def compute_rms(tensors: list[torch.Tensor | None]) -> torch.Tensor:
    present = [t for t in tensors if t is not None]
    if not present:
        return torch.tensor(0.0)
    square_sum = torch.stack([t.detach().float().square().sum() for t in present]).sum()
    count = sum(t.numel() for t in present)
    return (square_sum / count).sqrt()


def tensorstats(tensor: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}_mean": torch.mean(tensor),
        f"{prefix}_std": torch.std(tensor),
        f"{prefix}_min": torch.min(tensor),
        f"{prefix}_max": torch.max(tensor),
    }
