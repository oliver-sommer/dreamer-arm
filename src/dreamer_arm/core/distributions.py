"""Custom torch distributions used by the world model and actor-critic.

Includes:
- :class:`OneHotDist` — straight-through Gumbel-softmax discrete distribution
  with uniform mixing (``unimix_ratio``) for entropy regularisation.
- :class:`MultiOneHotDist` — product of independent ``OneHotDist`` factors
  (used for multi-discrete action / latent variables).
- :class:`TwoHot` — two-hot soft-target categorical loss over symexp-spaced
  bins (DreamerV3's reward / value parameterisation).
- :class:`MSEDist`, :class:`SymlogDist` — Gaussian-like log-prob heads with
  ``mse`` / ``abs`` aggregation modes.
- :class:`Bound` — wraps an unbounded distribution and projects its samples
  onto the unit ball by ``x / max(|x|, 1)``.

Plus factory helpers (``bounded_normal``, ``binary``, ``symexp_twohot``, …)
used as the ``MLPHead`` distribution constructors, and a discrete-KL helper.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import distributions as torchd
from torch.nn import functional as F

from dreamer_arm.utils.tensor import symexp, symlog, to_f32, to_i32

DistLike = Any  # heterogeneous return type for the factory functions below.


class OneHotDist(torchd.OneHotCategorical):
    """Categorical with optional uniform mixing and straight-through gradients."""

    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.0) -> None:
        probs = F.softmax(to_f32(logits), dim=-1)
        if unimix_ratio > 0.0:
            uniform = unimix_ratio / probs.shape[-1]
            probs = probs * (1.0 - unimix_ratio) + torch.full_like(probs, uniform)
            logits = torch.log(probs)
        super().__init__(logits=logits)

    @property
    def mode(self) -> torch.Tensor:
        argmax = torch.argmax(self.logits, dim=-1)
        one_hot = F.one_hot(argmax, self.logits.shape[-1]).to(self.logits.dtype)
        # Straight-through: forward = one-hot, backward = softmax logits.
        return one_hot.detach() + self.logits - self.logits.detach()

    def rsample(self, sample_shape: tuple[int, ...] = (), temperature: float = 1.0) -> torch.Tensor:  # type: ignore[override]
        return F.gumbel_softmax(self.logits, tau=temperature, hard=True, dim=-1)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError("Use rsample for straight-through samples.")


class MultiOneHotDist:
    """Product of independent :class:`OneHotDist`s with shape ``(..., sum(shape))``."""

    def __init__(self, logits: torch.Tensor, shape: tuple[int, ...], unimix_ratio: float = 0.0) -> None:
        self.shape = shape
        splits = torch.split(logits, list(shape), dim=-1)
        self.onehots: list[OneHotDist] = [OneHotDist(s, unimix_ratio=unimix_ratio) for s in splits]

    @property
    def mode(self) -> torch.Tensor:
        return torch.cat([d.mode for d in self.onehots], dim=-1)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        return torch.cat([d.rsample() for d in self.onehots], dim=-1)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        raise NotImplementedError("Use rsample.")

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        splits = torch.split(value, list(self.shape), dim=-1)
        # No explicit ``start=`` (a CPU scalar would break device-matching on CUDA);
        # ``shape`` is always non-empty so the int-0 default is never returned.
        # cast(): sum() is typed ``int | Tensor`` without a start, but at runtime
        # it always returns a Tensor here.
        return cast(torch.Tensor, sum(d.log_prob(s) for d, s in zip(self.onehots, splits, strict=True)))

    def entropy(self) -> torch.Tensor:
        return cast(torch.Tensor, sum(d.entropy() for d in self.onehots))


class TwoHot:
    """Two-hot soft-target categorical over a fixed bin grid.

    ``squash`` is applied to targets before localising them in bin space, and
    ``unsquash`` to the mode for output; defaults are the identity. The bins
    are typically constructed by :func:`symexp_twohot` (symexp-spaced).
    """

    def __init__(
        self,
        logits: torch.Tensor,
        bins: torch.Tensor,
        squash: Callable[[torch.Tensor], torch.Tensor] | None = None,
        unsquash: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.logits = to_f32(logits)
        assert self.logits.shape[-1] == bins.shape[-1], (self.logits.shape, bins.shape)
        self.bins = bins
        self.probs = F.softmax(self.logits, dim=-1)
        self.squash = squash if squash is not None else _identity
        self.unsquash = unsquash if unsquash is not None else _identity
        # log_prob() is called twice per instance on the value-loss path
        # (actor_critic.py); build the constant once instead of per call.
        self._one = torch.ones((), device=self.logits.device, dtype=torch.float32)

    def mode(self) -> torch.Tensor:
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1, p2, p3 = self.probs[..., :m], self.probs[..., m : m + 1], self.probs[..., m + 1 :]
            b1, b2, b3 = self.bins[..., :m], self.bins[..., m : m + 1], self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(dim=-1, keepdim=True) + ((p1 * b1).flip(dims=(-1,)) + (p3 * b3)).sum(
                dim=-1, keepdim=True
            )
        else:
            p1, p2 = self.probs[..., : n // 2], self.probs[..., n // 2 :]
            b1, b2 = self.bins[..., : n // 2], self.bins[..., n // 2 :]
            wavg = ((p1 * b1).flip(dims=(-1,)) + (p2 * b2)).sum(dim=-1, keepdim=True)
        return self.unsquash(wavg)

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        assert target.dtype == self.probs.dtype
        target_sq = self.squash(target.squeeze(-1)).detach()  # (...)
        below = to_i32(self.bins <= target_sq.unsqueeze(-1)).sum(dim=-1) - 1
        above = self.bins.shape[-1] - to_i32(self.bins > target_sq.unsqueeze(-1)).sum(dim=-1)
        below = below.clamp(0, self.bins.shape[-1] - 1)
        above = above.clamp(0, self.bins.shape[-1] - 1)
        equal = below == above
        d_below = torch.where(equal, self._one, (self.bins[below] - target_sq).abs())
        d_above = torch.where(equal, self._one, (self.bins[above] - target_sq).abs())
        total = d_below + d_above
        w_below = d_above / total
        w_above = d_below / total
        oh_below = to_f32(F.one_hot(below, num_classes=self.bins.shape[-1]))
        oh_above = to_f32(F.one_hot(above, num_classes=self.bins.shape[-1]))
        mixed = oh_below * w_below.unsqueeze(-1) + oh_above * w_above.unsqueeze(-1)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (mixed * log_pred).sum(dim=-1)


class MSEDist:
    """log_prob = -||mode - value||^2, aggregated by ``sum`` or ``mean`` over feature dims."""

    def __init__(self, mode: torch.Tensor, agg: str = "sum") -> None:
        self._mode = to_f32(mode)
        self._agg = agg

    def mode(self) -> torch.Tensor:
        return self._mode

    def mean(self) -> torch.Tensor:
        return self._mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        assert self._mode.dtype == value.dtype, (self._mode.dtype, value.dtype)
        distance = (self._mode - value) ** 2
        return -_aggregate(distance, self._agg)


class SymlogDist:
    """log_prob = -d(mode, symlog(value)) with mean/mode reported in original (symexp) space."""

    def __init__(
        self,
        mode: torch.Tensor,
        dist: str = "mse",
        agg: str = "sum",
        tol: float = 1e-8,
    ) -> None:
        self._mode = to_f32(mode)
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self) -> torch.Tensor:
        return symexp(self._mode)

    def mean(self) -> torch.Tensor:
        return symexp(self._mode)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        assert self._mode.shape == value.shape
        assert self._mode.dtype == value.dtype
        target = symlog(value)
        if self._dist == "mse":
            distance = (self._mode - target) ** 2
        elif self._dist == "abs":
            distance = torch.abs(self._mode - target)
        else:
            raise NotImplementedError(self._dist)
        distance = torch.where(distance < self._tol, torch.zeros_like(distance), distance)
        return -_aggregate(distance, self._agg)


class Bound:
    """Project a wrapped distribution's mean/sample to the unit ball ``x / max(|x|, 1)``."""

    def __init__(self, dist: Any) -> None:
        self._dist = dist

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    @property
    def mode(self) -> torch.Tensor:
        out = self._dist.mean
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def sample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        out = self._dist.rsample(sample_shape)
        return out / torch.clip(torch.abs(out), min=1.0).detach()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(x)


class TanhNormal:
    """Independent Normal followed by a differentiable tanh transform.

    The previous ``bounded_normal`` only bounded the Normal's *location* and
    relied on a later hard clamp for samples.  That made the distribution the
    actor scored differ from the distribution the world model and environment
    executed: all Normal tail mass collapsed onto atoms at -1/+1, while
    ``log_prob`` still evaluated the ordinary Normal density at those points.

    This wrapper keeps samples strictly inside the action bounds and applies
    the tanh change-of-variables correction in ``log_prob``.  ``pre_mean`` and
    ``pre_std`` intentionally expose the unsquashed parameters for policy-path
    diagnostics.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        self.pre_mean = to_f32(mean)
        self.pre_std = to_f32(std)
        self._base = torchd.Normal(self.pre_mean, self.pre_std)
        self._eps = eps

    @property
    def mode(self) -> torch.Tensor:
        return torch.tanh(self.pre_mean)

    @property
    def mean(self) -> torch.Tensor:
        # The exact transformed mean has no simple closed form.  The
        # deterministic action is the transformed base mode, which is also the
        # intended evaluation action.
        return self.mode

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        return torch.tanh(self._base.rsample(sample_shape))

    def sample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        return torch.tanh(self._base.sample(sample_shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # atanh is undefined at the closed interval endpoints.  Samples from
        # this distribution never reach them, but clamping makes scoring old
        # replay/checkpoint actions and defensive env-clamped values finite.
        bounded = value.clamp(-1.0 + self._eps, 1.0 - self._eps)
        pre = torch.atanh(bounded)
        log_det = torch.log1p(-(bounded * bounded) + self._eps)
        return (self._base.log_prob(pre) - log_det).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        # A tanh-Normal has no analytic entropy.  A reparameterised one-sample
        # estimate preserves the required entropy gradient without pretending
        # that the unsquashed Normal is the executed action distribution.
        sample = self.rsample()
        return -self.log_prob(sample)


def bounded_normal(x: torch.Tensor, min_std: float, max_std: float, **_: Any) -> DistLike:
    mean, std = torch.chunk(x, 2, dim=-1)
    std = (max_std - min_std) * torch.sigmoid(std + 2.0) + min_std
    return TanhNormal(to_f32(mean), to_f32(std))


def normal_std_fixed(mean: torch.Tensor, std: float, **_: Any) -> DistLike:
    base = torchd.Normal(to_f32(mean), torch.as_tensor(std, dtype=torch.float32))
    return Bound(torchd.Independent(base, 1))


def onehot(mean: torch.Tensor, unimix_ratio: float, **_: Any) -> OneHotDist:
    return OneHotDist(to_f32(mean), unimix_ratio=unimix_ratio)


def multi_onehot(mean: torch.Tensor, unimix_ratio: float, shape: tuple[int, ...], **_: Any) -> MultiOneHotDist:
    return MultiOneHotDist(to_f32(mean), shape, unimix_ratio=unimix_ratio)


def binary(logits: torch.Tensor, **_: Any) -> DistLike:
    base = torchd.Bernoulli(logits=to_f32(logits))
    return torchd.Independent(base, 1)


def symexp_twohot(logits: torch.Tensor, bin_num: int, **_: Any) -> TwoHot:
    """Two-hot over symexp-spaced bins from -symexp(20) to +symexp(20)."""
    if bin_num % 2 == 1:
        half = torch.linspace(-20, 0, (bin_num - 1) // 2 + 1, dtype=torch.float32, device=logits.device)
        half = symexp(half)
        bins = torch.cat([half, -half[:-1].flip(dims=(0,))], dim=0)
    else:
        # No exact middle bin for even counts; the previous mirror-and-flip
        # produced a duplicated, non-monotonic 0/-0. A single symmetric symexp
        # grid is strictly increasing and spans the full range.
        bins = symexp(torch.linspace(-20, 20, bin_num, dtype=torch.float32, device=logits.device))
    return TwoHot(to_f32(logits), bins)


def symlog_mse(logits: torch.Tensor, **_: Any) -> SymlogDist:
    return SymlogDist(to_f32(logits))


def mse(logits: torch.Tensor, **_: Any) -> MSEDist:
    return MSEDist(to_f32(logits))


def identity(logits: torch.Tensor, **_: Any) -> torch.Tensor:
    return logits


def kl(logits_left: torch.Tensor, logits_right: torch.Tensor) -> torch.Tensor:
    """KL(p_left || p_right) for categorical logits along the last axis."""
    log_p = torch.log_softmax(logits_left, dim=-1)
    log_q = torch.log_softmax(logits_right, dim=-1)
    p = torch.softmax(logits_left, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _aggregate(distance: torch.Tensor, agg: str) -> torch.Tensor:
    dims = list(range(distance.ndim))[2:]
    if agg == "mean":
        return distance.mean(dims) if dims else distance
    if agg == "sum":
        return distance.sum(dims) if dims else distance
    raise NotImplementedError(agg)
