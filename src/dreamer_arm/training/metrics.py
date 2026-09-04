"""The fixed W&B metric contract for online training and evaluation.

Model code intentionally returns richer diagnostics for tests and local
investigation. This module is the sole boundary that decides which scalar
series become permanent W&B keys and gives those keys a coherent hierarchy.
"""

from __future__ import annotations

from collections.abc import Mapping

# Raw model metric -> public W&B metric. Keeping this translation at the
# trainer boundary avoids coupling model implementations to one tracker.
CORE_TRAIN_METRICS = {
    # Replay batch health.
    "data/reward_mean": "train/data/reward_mean",
    "data/reward_std": "train/data/reward_std",
    "data/success_rate": "train/data/success_rate",
    "data/terminal_rate": "train/data/terminal_rate",
    "data/controller_clamp_rate": "train/data/controller_clamp_rate",
    "data/task_coverage": "train/data/task_coverage",
    "data/task_fraction_min": "train/data/task_fraction_min",
    "data/task_fraction_max": "train/data/task_fraction_max",
    # World model objectives and predictive skill.
    "loss/wm_total": "train/wm/loss_total",
    "loss/dyn": "train/wm/loss_dynamics",
    "loss/rep": "train/wm/loss_representation",
    "loss/barlow": "train/wm/loss_barlow",
    "loss/scene": "train/wm/loss_reconstruction_visual",
    "loss/proprio": "train/wm/loss_reconstruction_proprio",
    "loss/pred": "train/wm/loss_prediction",
    "loss/overshoot": "train/wm/loss_overshoot",
    "loss/rew": "train/wm/loss_reward",
    "loss/con": "train/wm/loss_continue",
    "loss/success": "train/wm/loss_success",
    "loss/constraint": "train/wm/loss_constraint",
    "pred/visual_mse": "train/wm/visual_mse",
    "pred/proprio_mse": "train/wm/proprio_mse",
    "pred/visual_skill_vs_persistence": "train/wm/visual_skill_vs_persistence",
    "pred/proprio_skill_vs_persistence": "train/wm/proprio_skill_vs_persistence",
    "rollout/visual_skill_h5": "train/wm/rollout_visual_skill_h5",
    "rollout/visual_skill_h10": "train/wm/rollout_visual_skill_h10",
    "rollout/proprio_skill_h5": "train/wm/rollout_proprio_skill_h5",
    "rollout/proprio_skill_h10": "train/wm/rollout_proprio_skill_h10",
    "reward/mae": "train/wm/reward_mae",
    "continue/brier": "train/wm/continue_brier",
    "success/target_rate": "train/wm/success_target_rate",
    "success/predicted_rate": "train/wm/success_predicted_rate",
    "success/brier": "train/wm/success_brier",
    "constraint/clamp_probability": "train/wm/constraint_probability",
    "constraint/clamp_rate": "train/wm/constraint_target_rate",
    "constraint/clamp_brier": "train/wm/constraint_brier",
    "constraint/retained_xyz_mae_m": "train/wm/retained_xyz_mae_m",
    "constraint/achieved_xyz_mae_m": "train/wm/achieved_xyz_mae_m",
    # Actor and imagination.
    "loss/policy": "train/actor/loss",
    "action_entropy": "train/actor/entropy",
    "actor/log_probability_mean": "train/actor/log_probability_mean",
    "actor/advantage_mean": "train/actor/advantage_mean",
    "actor/advantage_std": "train/actor/advantage_std",
    "actor/advantage_abs_mean": "train/actor/advantage_abs_mean",
    "actor/advantage_positive_fraction": "train/actor/advantage_positive_fraction",
    "ret": "train/actor/normalized_return_mean",
    "rew": "train/actor/imagination_reward",
    "shaped_rew": "train/actor/imagination_shaped_reward",
    "imag_success": "train/actor/imagination_success_probability",
    "imag_success_bonus": "train/actor/imagination_success_bonus",
    "imag_constraint_prob": "train/actor/imagination_constraint_probability",
    "imag_constraint_cost": "train/actor/imagination_constraint_cost",
    "con": "train/actor/imagination_continue_probability",
    "weight": "train/actor/imagination_weight",
    "action_xyz_near_bound_fraction": "train/actor/action_xyz_near_bound_fraction",
    "action_gripper_mean": "train/actor/action_gripper_mean",
    "actor/policy_std_mean": "train/actor/policy_std_mean",
    # Critic agreement with imagined and replay lambda returns.
    "loss/value": "train/critic/loss_imagined",
    "loss/repval": "train/critic/loss_replay",
    "critic/value_mean": "train/critic/value_mean",
    "critic/target_mean": "train/critic/target_mean",
    "critic/value_bias": "train/critic/value_bias",
    "critic/value_mae": "train/critic/value_mae",
    "critic/value_rmse": "train/critic/value_rmse",
    "critic/explained_variance": "train/critic/explained_variance",
    "critic/replay_value_mean": "train/critic/replay_value_mean",
    "critic/replay_target_mean": "train/critic/replay_target_mean",
    "critic/replay_value_mae": "train/critic/replay_value_mae",
    "critic/replay_explained_variance": "train/critic/replay_explained_variance",
    "critic/slow_value_mean": "train/critic/slow_value_mean",
    "critic/slow_value_gap_mae": "train/critic/slow_value_gap_mae",
    # Optimizer essentials.
    "opt/loss": "train/optimizer/loss_total",
    "opt/lr": "train/optimizer/learning_rate",
    "opt/grad_skipped": "train/optimizer/grad_skipped_rate",
}

DIAGNOSTIC_TRAIN_METRICS = {
    "rollout/action_zero_excess_mse": "train/wm/action_zero_excess_mse",
    "rollout/action_shuffled_excess_mse": "train/wm/action_shuffled_excess_mse",
    "ret_scale_mean": "train/actor/return_scale_mean",
    "ret_scale_min": "train/actor/return_scale_min",
    "ret_scale_max": "train/actor/return_scale_max",
    "opt/grad_norm_before_clip": "train/optimizer/gradient_norm_before_clip",
    "opt/grad_norm_after_clip": "train/optimizer/gradient_norm_after_clip",
    "opt/param_rms": "train/optimizer/parameter_rms",
    "opt/update_rms": "train/optimizer/update_rms",
    "opt/update_to_param_ratio": "train/optimizer/update_to_parameter_ratio",
    "opt/grad_scale": "train/optimizer/amp_scale",
    "dyn_entropy": "train/wm/dynamics_entropy",
    "rep_entropy": "train/wm/representation_entropy",
    "barlow/invariance": "train/wm/barlow_invariance",
    "barlow/redundancy": "train/wm/barlow_redundancy",
}

for _axis in ("workspace", "lag", "joint_limit"):
    for _raw_suffix, _public_suffix in (
        ("prob", "probability"),
        ("rate", "target_rate"),
        ("precision", "precision"),
        ("recall", "recall"),
    ):
        DIAGNOSTIC_TRAIN_METRICS[f"constraint/{_axis}_{_raw_suffix}"] = f"train/wm/constraint_{_axis}_{_public_suffix}"

TRAIN_HISTOGRAMS = {
    "diagnostic/advantage": "train/actor/advantage_histogram",
    "diagnostic/action": "train/actor/action_histogram",
    "diagnostic/value_error": "train/critic/value_error_histogram",
    "diagnostic/replay_value_error": "train/critic/replay_value_error_histogram",
}

EVAL_METRICS = {
    "eval/success_mean": "eval/success_mean",
    "eval/success_min": "eval/success_min",
    "eval/success_std": "eval/success_std",
    "eval/return_mean": "eval/return_mean",
    "eval/return_min": "eval/return_min",
    "eval/return_std": "eval/return_std",
    "eval/episodes_completed": "eval/episodes_completed",
    "eval/episodes_per_task": "eval/episodes_per_task",
    "eval/action_xyz_near_bound_fraction": "eval/action_xyz_near_bound_fraction",
    "eval/action_gripper_mean": "eval/action_gripper_mean",
    "eval/action_task_id_sensitivity": "eval/action_task_id_sensitivity",
    "eval/action_proprio_sensitivity": "eval/action_proprio_sensitivity",
    "eval/action_visual_sensitivity": "eval/action_visual_sensitivity",
    "eval/predicted_success_probability": "eval/predicted_success_probability",
    "eval/predicted_success_brier": "eval/predicted_success_brier",
    "eval/pred_constraint_brier": "eval/pred_constraint_brier",
    "eval/pred_constraint_retained_mae_m": "eval/pred_constraint_retained_mae_m",
    "eval/pred_constraint_achieved_mae_m": "eval/pred_constraint_achieved_mae_m",
    "eval/ctrl_frac_ws_clamped": "eval/controller_workspace_clamp_rate",
    "eval/ctrl_frac_lag_clamped": "eval/controller_lag_clamp_rate",
    "eval/ctrl_frac_joint_limit_clamped": "eval/controller_joint_limit_clamp_rate",
    "eval/ctrl_frac_undertracking": "eval/controller_undertracking_rate",
}


def compact_train_metrics[Value](metrics: Mapping[str, Value], *, diagnostics: bool = False) -> dict[str, Value]:
    """Translate detailed model output into the fixed W&B scalar contract."""

    contract = CORE_TRAIN_METRICS | (DIAGNOSTIC_TRAIN_METRICS if diagnostics else {})
    return {public: metrics[raw] for raw, public in contract.items() if raw in metrics}


def compact_train_histograms[Value](metrics: Mapping[str, Value]) -> dict[str, Value]:
    """Select bounded diagnostic tensors that should become W&B histograms."""

    return {public: metrics[raw] for raw, public in TRAIN_HISTOGRAMS.items() if raw in metrics}


def compact_eval_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Keep unified eval aggregates and chartable per-task success curves."""

    selected = {public: metrics[raw] for raw, public in EVAL_METRICS.items() if raw in metrics}
    selected.update({name: value for name, value in metrics.items() if name.startswith("eval/success/")})
    return selected


__all__ = [
    "CORE_TRAIN_METRICS",
    "DIAGNOSTIC_TRAIN_METRICS",
    "EVAL_METRICS",
    "TRAIN_HISTOGRAMS",
    "compact_eval_metrics",
    "compact_train_histograms",
    "compact_train_metrics",
]
