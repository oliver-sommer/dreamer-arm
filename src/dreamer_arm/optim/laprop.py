"""LaProp optimizer.

Original implementation © 2020 Wang, T. Zhikang (MIT licence); ported here
with PyTorch 2.x-compatible call signatures for in-place tensor ops
(`addcmul_(t1, t2, value=alpha)` rather than the removed positional form).

Reference: "LaProp: Separating Momentum and Adaptivity in Adam"
(https://github.com/Z-T-WANG/LaProp-Optimizer).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class LaProp(Optimizer):
    """LaProp optimiser. Same API surface as :class:`torch.optim.Adam`."""

    steps_before_using_centered: int = 10

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 4e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-15,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        centered: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults: dict[str, Any] = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            centered=centered,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad: bool = group["amsgrad"]
            centered: bool = group["centered"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients.")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_lr_1"] = 0.0
                    state["exp_avg_lr_2"] = 0.0
                    if centered:
                        state["exp_mean_avg_beta2"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                # m_2 <- beta2 * m_2 + (1 - beta2) * g^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Effective bias-correction weights track an EMA over learning
                # rates so that schedule changes are handled smoothly.
                state["exp_avg_lr_1"] = state["exp_avg_lr_1"] * beta1 + (1 - beta1) * lr
                state["exp_avg_lr_2"] = state["exp_avg_lr_2"] * beta2 + (1 - beta2)

                bc1 = state["exp_avg_lr_1"] / lr if lr != 0.0 else 1.0
                step_size = 1.0 / bc1
                bc2 = state["exp_avg_lr_2"]

                denom = exp_avg_sq
                if centered:
                    state["exp_mean_avg_beta2"].mul_(beta2).add_(grad, alpha=1 - beta2)
                    if state["step"] > self.steps_before_using_centered:
                        denom = denom - state["exp_mean_avg_beta2"] ** 2

                if amsgrad and not (centered and state["step"] <= self.steps_before_using_centered):
                    torch.max(state["max_exp_avg_sq"], denom, out=state["max_exp_avg_sq"])
                    denom = state["max_exp_avg_sq"]

                denom = denom.div(bc2).sqrt_().add_(eps)
                # m_1 <- beta1 * m_1 + (1 - beta1) * lr * g / sqrt(denom)
                exp_avg.mul_(beta1).addcdiv_(grad, denom, value=(1 - beta1) * lr)

                p.add_(exp_avg, alpha=-step_size)
                if wd != 0.0:
                    p.add_(p, alpha=-wd)
