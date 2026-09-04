"""Tests for `OptimStep`: LaProp + AGC + non-finite guard + LR warmup, bundled."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from dreamer_arm.core.optim.step import OptimStep


def _cfg(**overrides: Any) -> SimpleNamespace:
    defaults: dict[str, Any] = dict(  # noqa: C408
        lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, agc=0.3, pmin=1e-3, log_grads=False, warmup=0
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _linear_and_named() -> tuple[nn.Linear, dict[str, nn.Parameter]]:
    torch.manual_seed(0)
    lin = nn.Linear(2, 2)
    return lin, dict(lin.named_parameters())


def test_step_applies_a_normal_gradient() -> None:
    lin, named = _linear_and_named()
    step = OptimStep(named, _cfg(), torch.device("cpu"))
    before = [p.clone() for p in named.values()]

    step.backward(lin(torch.randn(4, 2)).sum())
    mets = step.step()

    assert step.stepped
    assert mets["opt/grad_skipped"].item() == 0.0
    assert any(not torch.equal(b, p) for b, p in zip(before, named.values(), strict=True))


def test_step_skips_and_leaves_params_unchanged_on_non_finite_gradient() -> None:
    lin, named = _linear_and_named()
    step = OptimStep(named, _cfg(), torch.device("cpu"))
    before = [p.clone() for p in named.values()]

    step.backward(lin(torch.randn(4, 2)).sum() * float("inf"))
    mets = step.step()

    assert not step.stepped
    assert mets["opt/grad_skipped"].item() == 1.0
    assert all(torch.equal(b, p) for b, p in zip(before, named.values(), strict=True))


def test_step_lr_ramps_up_during_warmup() -> None:
    lin, named = _linear_and_named()
    step = OptimStep(named, _cfg(warmup=10, lr=1.0), torch.device("cpu"))

    step.backward(lin(torch.randn(4, 2)).sum())
    mets = step.step()
    assert 0.0 < mets["opt/lr"].item() < 1.0  # first of 10 warmup steps


def test_step_state_dict_round_trips() -> None:
    lin, named = _linear_and_named()
    step = OptimStep(named, _cfg(), torch.device("cpu"))
    step.backward(lin(torch.randn(4, 2)).sum())
    step.step()

    state = step.state_dict()
    step2 = OptimStep(named, _cfg(), torch.device("cpu"))
    step2.load_state_dict(state)  # must not raise


def test_diagnostic_step_reports_gradient_and_update_health() -> None:
    lin, named = _linear_and_named()
    step = OptimStep(named, _cfg(), torch.device("cpu"))
    step.backward(lin(torch.randn(4, 2)).sum())

    mets = step.step(diagnostics=True)

    for key in (
        "opt/grad_norm_before_clip",
        "opt/grad_norm_after_clip",
        "opt/param_rms",
        "opt/update_rms",
        "opt/update_to_param_ratio",
    ):
        assert torch.isfinite(mets[key])
    assert mets["opt/grad_norm_after_clip"] <= mets["opt/grad_norm_before_clip"]
    assert "opt/grad_scale" not in mets  # CPU has no active AMP scaler.
