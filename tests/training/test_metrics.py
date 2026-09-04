"""Tests for the compact W&B metric contract."""

from __future__ import annotations

import torch

from dreamer_arm.training.metrics import compact_eval_metrics, compact_train_histograms, compact_train_metrics


def test_compact_train_metrics_keeps_dashboard_signals_only() -> None:
    metrics = {
        "loss/pred": torch.tensor(1.0),
        "action_entropy": torch.tensor(-2.0),
        "rollout/visual_skill_h10": torch.tensor(0.2),
        "ret_005_task_0": torch.tensor(-3.0),
        "action_x_pre_mean": torch.tensor(4.0),
    }

    compact = compact_train_metrics(metrics)

    assert set(compact) == {
        "train/wm/loss_prediction",
        "train/actor/entropy",
        "train/wm/rollout_visual_skill_h10",
    }


def test_compact_eval_metrics_keeps_task_outcomes_not_forensics() -> None:
    metrics = {
        "eval/success_mean": 0.3,
        "eval/success/reach_v3": 1.0,
        "eval/return/reach_v3": 12.0,
        "eval/ctrl_frac_lag_clamped": 0.2,
        "eval/ctrl_frac_lag_clamped/reach_v3": 0.3,
        "eval/action_x_mean/reach_v3": 0.4,
        "eval/actor_param_checksum": 123.0,
    }

    compact = compact_eval_metrics(metrics)

    assert compact == {
        "eval/success_mean": 0.3,
        "eval/success/reach_v3": 1.0,
        "eval/controller_lag_clamp_rate": 0.2,
    }


def test_diagnostics_are_cadenced_and_histograms_are_explicit() -> None:
    metrics = {
        "actor/advantage_mean": torch.tensor(0.25),
        "ret_scale_max": torch.tensor(4.0),
        "opt/grad_norm_before_clip": torch.tensor(3.0),
        "diagnostic/advantage": torch.arange(4),
        "ret_005_task_0": torch.tensor(-2.0),
    }

    assert compact_train_metrics(metrics) == {"train/actor/advantage_mean": torch.tensor(0.25)}
    assert set(compact_train_metrics(metrics, diagnostics=True)) == {
        "train/actor/advantage_mean",
        "train/actor/return_scale_max",
        "train/optimizer/gradient_norm_before_clip",
    }
    assert set(compact_train_histograms(metrics)) == {"train/actor/advantage_histogram"}
