"""Tests for the compact W&B metric contract."""

from __future__ import annotations

import torch

from dreamer_arm.training.metrics import compact_eval_metrics, compact_train_metrics


def test_compact_train_metrics_keeps_dashboard_signals_only() -> None:
    metrics = {
        "loss/pred": torch.tensor(1.0),
        "action_entropy": torch.tensor(-2.0),
        "rollout/visual_skill_h10": torch.tensor(0.2),
        "ret_005_task_0": torch.tensor(-3.0),
        "action_x_pre_mean": torch.tensor(4.0),
    }

    compact = compact_train_metrics(metrics)

    assert set(compact) == {"loss/pred", "action_entropy", "rollout/visual_skill_h10"}


def test_compact_eval_metrics_keeps_task_outcomes_not_forensics() -> None:
    metrics = {
        "eval/success_mean": 0.3,
        "eval/success/reach_v3": 1.0,
        "eval/return/reach_v3": 12.0,
        "eval/ctrl_frac_lag_clamped": 0.2,
        "eval/ctrl_frac_lag_clamped/reach_v3": 0.3,
        "eval/action_x_mean/reach_v3": 0.4,
        "eval/actor_param_checksum": 123.0,
        "eval_robust/success_mean": 0.8,
        "eval_robust/success/reach_v3": 1.0,
    }

    compact = compact_eval_metrics(metrics)

    assert compact == {
        "eval/success_mean": 0.3,
        "eval/success/reach_v3": 1.0,
        "eval/return/reach_v3": 12.0,
        "eval/ctrl_frac_lag_clamped": 0.2,
        "eval_robust/success_mean": 0.8,
        "eval_robust/success/reach_v3": 1.0,
    }
