"""One training step's optimiser plumbing: LaProp + AGC + non-finite guard + LR warmup.

Bundled into a single object so :meth:`~dreamer_arm.core.model.Dreamer.update`
reads as "compute the loss, then take a step" instead of interleaving that
with unscale / clip / guard / schedule bookkeeping.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import LambdaLR

from dreamer_arm.core.optim.agc import adaptive_grad_clip
from dreamer_arm.core.optim.laprop import LaProp
from dreamer_arm.utils.tensor import compute_global_norm, compute_rms


class OptimStep:
    """A LaProp optimiser plus its scaler / LR schedule / AGC over a fixed parameter set."""

    def __init__(self, named_params: dict[str, nn.Parameter], config: Any, device: torch.device) -> None:
        self._named_params = named_params
        self.device = device
        self._optimizer = LaProp(
            list(named_params.values()),
            lr=float(config.lr),
            betas=(float(config.beta1), float(config.beta2)),
            eps=float(config.eps),
        )
        self.amp_enabled = device.type == "cuda"
        self._scaler = GradScaler(device=device.type, enabled=self.amp_enabled)
        self._agc_clip = float(config.agc)
        self._agc_pmin = float(config.pmin)
        self._log_grads = bool(config.log_grads)
        self.stepped = False

        warmup = int(config.warmup)

        def _lr_lambda(step: int) -> float:
            return min(1.0, (step + 1) / warmup) if warmup else 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=_lr_lambda)

    def backward(self, loss: torch.Tensor) -> None:
        self._scaler.scale(loss).backward()

    def step(self) -> dict[str, torch.Tensor]:
        """Unscale, AGC-clip, guard against non-finite gradients, step, schedule."""
        mets: dict[str, torch.Tensor] = {}
        self._scaler.unscale_(self._optimizer)
        params = list(self._named_params.values())

        if self._log_grads:
            old_params = [p.data.clone().detach() for p in params]
            grads = [p.grad for p in params if p.grad is not None]
            mets["opt/grad_norm"] = compute_global_norm(grads)
            mets["opt/grad_rms"] = compute_rms(grads)

        adaptive_grad_clip(params, self._agc_clip, self._agc_pmin)

        # Non-finite gradient guard.  A single NaN/inf gradient must never reach
        # the optimizer: without GradScaler (CUDA-only here), a disabled scaler
        # steps unconditionally, so one transient NaN (e.g. an MPS sampling
        # glitch) would corrupt every weight permanently.  On CUDA the scaler
        # already skips such steps and lowers its scale; off-CUDA we check the
        # (already-unscaled) grads ourselves.
        if self.amp_enabled:
            scale_before = self._scaler.get_scale()
            self._scaler.step(self._optimizer)
            self._scaler.update()
            stepped = self._scaler.get_scale() >= scale_before
        else:
            stepped = all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            if stepped:
                self._optimizer.step()
        # Only advance the LR schedule when the optimizer actually ran.
        if stepped:
            self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        self.stepped = stepped

        # device= matters, not just style: these land in the same metrics dict
        # as the wm/ac losses (computed on `device`), and WandbLogger.scalars()
        # stacks every tensor value in one call -- a bare torch.tensor(x) here
        # defaults to CPU and torch.stack refuses to mix devices.
        mets["opt/grad_skipped"] = torch.tensor(0.0 if stepped else 1.0, device=self.device)
        mets["opt/lr"] = torch.tensor(self._scheduler.get_last_lr()[0], device=self.device)
        mets["opt/grad_scale"] = torch.tensor(self._scaler.get_scale(), device=self.device)
        if self._log_grads:
            updates = [(p.data - old) for p, old in zip(params, old_params, strict=True)]
            mets["opt/param_rms"] = compute_rms([p.data for p in params])
            mets["opt/update_rms"] = compute_rms(updates)
        return mets

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self._optimizer.state_dict(),
            "scaler": self._scaler.state_dict(),
            "scheduler": self._scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state["optimizer"])
        self._scaler.load_state_dict(state["scaler"])
        self._scheduler.load_state_dict(state["scheduler"])


__all__ = ["OptimStep"]
