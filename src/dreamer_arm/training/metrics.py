"""Compact W&B metric contracts for online training.

Model and evaluation code intentionally expose richer diagnostic dictionaries
for tests and one-off investigations.  Sending every diagnostic to W&B turns
MT10 into hundreds of permanent scalar series, though, because many families
are repeated per task.  The trainer uses the selectors here as the single
boundary between detailed internal diagnostics and the small operational
dashboard used for normal runs.
"""

from __future__ import annotations

from collections.abc import Mapping

TRAIN_METRICS = frozenset(
    {
        # Optimisation objectives.  Include every supported WM variant.
        "loss/dyn",
        "loss/rep",
        "loss/pred",
        "loss/overshoot",
        "loss/constraint",
        "loss/rew",
        "loss/con",
        "loss/success",
        "loss/policy",
        "loss/value",
        "loss/repval",
        # Optimiser/replay health.
        "opt/lr",
        "opt/grad_skipped",
        # One-step and useful open-loop WM skill summaries.
        "pred/visual_skill_vs_persistence",
        "pred/proprio_skill_vs_persistence",
        "rollout/visual_skill_h5",
        "rollout/proprio_skill_h5",
        "rollout/visual_skill_h10",
        "rollout/proprio_skill_h10",
        # Controller-model calibration and motion accuracy.
        "constraint/clamp_probability",
        "constraint/clamp_rate",
        "constraint/clamp_brier",
        "constraint/retained_xyz_mae_m",
        "constraint/achieved_xyz_mae_m",
        # Success calibration and actor behaviour.
        "success/target_rate",
        "success/predicted_rate",
        "action_entropy",
        "action_xyz_near_bound_fraction",
        "rew",
        "shaped_rew",
        "imag_success_bonus",
        "imag_constraint_cost",
    }
)


EVAL_METRICS = frozenset(
    {
        "eval/success_mean",
        "eval/return_mean",
        "eval/episodes_completed",
        "eval/action_xyz_near_bound_fraction",
        "eval/action_gripper_mean",
        "eval/predicted_success_probability",
        "eval/pred_constraint_brier",
        "eval/pred_constraint_retained_mae_m",
        "eval/pred_constraint_achieved_mae_m",
        "eval/ctrl_frac_ws_clamped",
        "eval/ctrl_frac_lag_clamped",
        "eval/ctrl_frac_joint_limit_clamped",
        "eval/ctrl_frac_undertracking",
    }
)

ROBUST_EVAL_METRICS = frozenset(name.replace("eval/", "eval_robust/", 1) for name in EVAL_METRICS)


def compact_train_metrics[Value](metrics: Mapping[str, Value]) -> dict[str, Value]:
    """Keep the fixed training dashboard contract from a detailed update."""

    return {name: value for name, value in metrics.items() if name in TRAIN_METRICS}


def compact_eval_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Keep aggregate eval health plus per-task success and return curves."""

    return {
        name: value
        for name, value in metrics.items()
        if name in EVAL_METRICS
        or name in ROBUST_EVAL_METRICS
        or name.startswith(("eval/success/", "eval/return/", "eval_robust/success/", "eval_robust/return/"))
    }


__all__ = [
    "EVAL_METRICS",
    "ROBUST_EVAL_METRICS",
    "TRAIN_METRICS",
    "compact_eval_metrics",
    "compact_train_metrics",
]
