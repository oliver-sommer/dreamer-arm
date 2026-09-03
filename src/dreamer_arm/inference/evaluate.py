"""Deterministic policy evaluation: rollouts with ``eval_mode=True``, no learning.

:func:`evaluate` is pure — it returns metrics and an optional video instead of
writing to a logger — so the same code serves the training loop's periodic eval
and the standalone entrypoint below.

Evaluation always resets with a fixed seed (:data:`EVAL_SEED`), so every eval
pass samples the same task instances and camera poses.  Success rates are then
comparable across checkpoints instead of being part reset noise.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from dreamer_arm.utils.logging import phase

log = logging.getLogger(__name__)

#: Fixed seed for evaluation resets, so eval is deterministic across checkpoints.
EVAL_SEED = 12345
EVAL_SEEDS = (12345, 23456, 34567, 45678, 56789)
ACTION_TRACE_STEPS = 20
_ACTION_LABELS = ("x", "y", "z", "gripper")


@dataclass
class ActionTrace:
    """Tabular deterministic action trace for one evaluation pass."""

    columns: list[str] = field(default_factory=list)
    rows: list[list[str | int | float]] = field(default_factory=list)


@dataclass
class EvalResult:
    """Outcome of one evaluation pass.

    Attributes:
        metrics: Ready-to-log scalars, already namespaced under ``eval/``.
        video:   ``(T, H, W, C)`` frames from env 0's first episode, or
                 ``None`` when the envs expose no ``scene`` observation.
        action_trace: One compact table containing the first deterministic
                      actions for every task, rather than hundreds of scalar
                      series that W&B renders as separate charts.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    video: np.ndarray | None = None
    action_trace: ActionTrace | None = None


def evaluate(
    agent: Any,
    envs: Any,
    episodes: int,
    *,
    seeds: tuple[int, ...] | None = None,
    capture_artifacts: bool = True,
) -> EvalResult:
    """Run at least ``episodes`` evaluation episodes and summarise them.

    Episodes are collected in parallel across the vector env, so the effective
    count is rounded up to a whole number of rounds across ``envs.num_envs``.

    The env is reset with a fixed seed. A caller sharing its training env must
    reset it again before resuming collection.
    """
    n = envs.num_envs
    device = agent.device
    obs_keys = sorted(envs.observation_space.spaces.keys())
    num_rounds = max(1, math.ceil(episodes / n))
    eval_seeds = seeds or tuple(
        EVAL_SEEDS[index] if index < len(EVAL_SEEDS) else EVAL_SEED + 1009 * index for index in range(num_rounds)
    )
    if len(eval_seeds) < num_rounds:
        raise ValueError(f"evaluation needs {num_rounds} seeds, got {len(eval_seeds)}")

    task_success: dict[str, list[float]] = {}
    task_returns: dict[str, list[float]] = {}
    # YamArm controller diagnostics, averaged across eval episodes (YAM only).
    ctrl_diags: dict[str, list[float]] = {}
    ctrl_diags_by_task: dict[str, dict[str, list[float]]] = {}
    reward_diags: dict[str, list[float]] = {}

    obs_np = envs.reset(seed=eval_seeds[0])
    state = agent.get_initial_state(n)
    is_first = np.ones(n, dtype=bool)
    completed = np.zeros(n, dtype=np.int32)  # episodes done per env slot
    episode_steps = np.zeros(n, dtype=np.int32)
    episode_returns = np.zeros(n, dtype=np.float64)

    # The first episode in every pinned env slot supplies the requested
    # task-by-task deterministic action trace.  Slot names become available in
    # final_info at episode end, so retain by slot until then.
    slot_actions: list[list[np.ndarray]] = [[] for _ in range(n)]
    slot_pre_means: list[list[np.ndarray]] = [[] for _ in range(n)]
    slot_pre_stds: list[list[np.ndarray]] = [[] for _ in range(n)]
    slot_task_names: list[str | None] = [None] * n
    all_actions: list[np.ndarray] = []
    all_pre_means: list[np.ndarray] = []
    all_pre_stds: list[np.ndarray] = []
    all_task_sensitivity: list[np.ndarray] = []
    all_proprio_sensitivity: list[np.ndarray] = []
    all_visual_sensitivity: list[np.ndarray] = []
    all_success_probability: list[np.ndarray] = []
    all_constraint_probability: list[np.ndarray] = []
    all_constraint_target: list[np.ndarray] = []
    all_retained_prediction: list[np.ndarray] = []
    all_retained_target: list[np.ndarray] = []
    all_achieved_prediction: list[np.ndarray] = []
    all_achieved_target: list[np.ndarray] = []
    slot_all_actions: list[list[np.ndarray]] = [[] for _ in range(n)]
    slot_task_sensitivity: list[list[float]] = [[] for _ in range(n)]
    slot_proprio_sensitivity: list[list[float]] = [[] for _ in range(n)]
    slot_visual_sensitivity: list[list[float]] = [[] for _ in range(n)]
    task_ids_seen: set[int] = set()
    task_id_rows = 0
    task_id_valid_rows = 0

    video_frames: list[np.ndarray] = []
    video_done = False
    seeded_round = 0

    while completed.min() < num_rounds:
        if capture_artifacts and not video_done and "scene" in obs_np:
            video_frames.append(obs_np["scene"][0])

        obs_torch: dict[str, torch.Tensor] = {
            k: torch.from_numpy(obs_np[k]).to(device, non_blocking=True) for k in obs_keys
        }
        obs_torch["is_first"] = torch.from_numpy(is_first).to(device, non_blocking=True)

        if "task_id" in obs_np:
            task_id = np.asarray(obs_np["task_id"])
            row_sums = task_id.sum(axis=-1)
            valid = np.isfinite(task_id).all(axis=-1) & np.isclose(row_sums, 1.0) & (task_id >= 0.0).all(axis=-1)
            task_id_rows += int(task_id.shape[0])
            task_id_valid_rows += int(valid.sum())
            task_ids_seen.update(int(index) for index in task_id[valid].argmax(axis=-1))

        with torch.no_grad():
            action_t, next_state = agent.act(obs_torch, state, eval_mode=True)
        action_np = action_t.detach().cpu().numpy()
        all_actions.append(action_np.copy())

        policy_diag: dict[str, torch.Tensor] = {}
        if hasattr(agent, "policy_diagnostics"):
            policy_diag = agent.policy_diagnostics(next_state)
        pre_mean_np = policy_diag.get("pre_mean")
        pre_std_np = policy_diag.get("pre_std")
        task_sensitivity = policy_diag.get("task_id_action_sensitivity")
        proprio_sensitivity = policy_diag.get("proprio_action_sensitivity")
        visual_sensitivity = policy_diag.get("visual_action_sensitivity")
        success_probability = policy_diag.get("success_probability")
        constraint_probability = policy_diag.get("constraint_probability")
        retained_prediction = policy_diag.get("constraint_retained_xyz")
        achieved_prediction = policy_diag.get("constraint_achieved_xyz")
        pre_mean_arr = pre_mean_np.detach().cpu().numpy() if pre_mean_np is not None else None
        pre_std_arr = pre_std_np.detach().cpu().numpy() if pre_std_np is not None else None
        task_sensitivity_arr = task_sensitivity.detach().cpu().numpy() if task_sensitivity is not None else None
        proprio_sensitivity_arr = (
            proprio_sensitivity.detach().cpu().numpy() if proprio_sensitivity is not None else None
        )
        visual_sensitivity_arr = visual_sensitivity.detach().cpu().numpy() if visual_sensitivity is not None else None
        success_probability_arr = (
            success_probability.detach().cpu().numpy() if success_probability is not None else None
        )
        constraint_probability_arr = (
            constraint_probability.detach().cpu().numpy() if constraint_probability is not None else None
        )
        retained_prediction_arr = (
            retained_prediction.detach().cpu().numpy() if retained_prediction is not None else None
        )
        achieved_prediction_arr = (
            achieved_prediction.detach().cpu().numpy() if achieved_prediction is not None else None
        )
        if pre_mean_arr is not None:
            all_pre_means.append(pre_mean_arr.copy())
        if pre_std_arr is not None:
            all_pre_stds.append(pre_std_arr.copy())
        if task_sensitivity_arr is not None:
            all_task_sensitivity.append(task_sensitivity_arr.copy())
        if proprio_sensitivity_arr is not None:
            all_proprio_sensitivity.append(proprio_sensitivity_arr.copy())
        if visual_sensitivity_arr is not None:
            all_visual_sensitivity.append(visual_sensitivity_arr.copy())
        if success_probability_arr is not None:
            all_success_probability.append(success_probability_arr.copy())

        for i in range(n):
            if completed[i] < num_rounds:
                slot_all_actions[i].append(action_np[i].copy())
                if task_sensitivity_arr is not None:
                    slot_task_sensitivity[i].append(float(task_sensitivity_arr[i]))
                if proprio_sensitivity_arr is not None:
                    slot_proprio_sensitivity[i].append(float(proprio_sensitivity_arr[i]))
                if visual_sensitivity_arr is not None:
                    slot_visual_sensitivity[i].append(float(visual_sensitivity_arr[i]))
            if capture_artifacts and completed[i] == 0 and episode_steps[i] < ACTION_TRACE_STEPS:
                slot_actions[i].append(action_np[i].copy())
                if pre_mean_arr is not None:
                    slot_pre_means[i].append(pre_mean_arr[i].copy())
                if pre_std_arr is not None:
                    slot_pre_stds[i].append(pre_std_arr[i].copy())

        obs_next_np, rewards, terms, truncs, info = envs.step(action_np)
        transition_info = info.get("transition", {})
        ctrl_valid = np.asarray(transition_info.get("ctrl_valid", np.zeros(n, dtype=bool)), dtype=bool)
        ctrl_target = np.asarray(transition_info.get("ctrl_clamp", np.zeros((n, 3))), dtype=np.float32)
        retained_target = np.asarray(transition_info.get("ctrl_retained_xyz", np.zeros((n, 3))), dtype=np.float32)
        achieved_target = np.asarray(transition_info.get("ctrl_achieved_xyz", np.zeros((n, 3))), dtype=np.float32)
        if constraint_probability_arr is not None and ctrl_valid.shape == (n,) and ctrl_target.shape == (n, 3):
            all_constraint_probability.append(constraint_probability_arr[ctrl_valid].copy())
            all_constraint_target.append(ctrl_target[ctrl_valid].copy())
        if retained_prediction_arr is not None and retained_target.shape == (n, 3) and ctrl_valid.shape == (n,):
            all_retained_prediction.append(retained_prediction_arr[ctrl_valid].copy())
            all_retained_target.append(retained_target[ctrl_valid].copy())
        if achieved_prediction_arr is not None and achieved_target.shape == (n, 3) and ctrl_valid.shape == (n,):
            all_achieved_prediction.append(achieved_prediction_arr[ctrl_valid].copy())
            all_achieved_target.append(achieved_target[ctrl_valid].copy())
        episode_returns += rewards
        episode_steps += 1
        done = terms | truncs

        for i in range(n):
            if done[i] and completed[i] < num_rounds:
                fin = info["final_info"][i]
                task_name = fin.get("task_name", f"env_{i}") if fin is not None else f"env_{i}"
                success = float(fin.get("success", 0.0)) if fin is not None else 0.0
                task_success.setdefault(task_name, []).append(success)
                task_returns.setdefault(task_name, []).append(float(episode_returns[i]))
                slot_task_names[i] = task_name
                if fin is not None:
                    for k, v in fin.get("ctrl_diag", {}).items():
                        ctrl_diags.setdefault(k, []).append(float(v))
                        ctrl_diags_by_task.setdefault(task_name, {}).setdefault(k, []).append(float(v))
                    for k, v in fin.get("reward_diag", {}).items():
                        reward_diags.setdefault(k, []).append(float(v))
                completed[i] += 1
                episode_returns[i] = 0.0
                episode_steps[i] = 0

        if not video_done and done[0]:
            video_done = True

        is_first = done.copy()
        state = next_state
        if done.any():
            done_t = torch.from_numpy(done).to(device)
            state["prev_action"][done_t] = 0.0

        completed_round = int(completed.min())
        if completed_round > seeded_round and completed_round < num_rounds:
            seeded_round = completed_round
            obs_np = envs.reset(seed=eval_seeds[seeded_round])
            state = agent.get_initial_state(n)
            is_first = np.ones(n, dtype=bool)
            episode_returns[:] = 0.0
            episode_steps[:] = 0
        else:
            obs_np = obs_next_np

    metrics: dict[str, float] = {}
    all_successes: list[float] = []
    for task_name, successes in task_success.items():
        safe_name = task_name.replace("-", "_").replace(" ", "_")
        metrics[f"eval/success/{safe_name}"] = float(np.mean(successes))
        all_successes.extend(successes)

    if all_successes:
        metrics["eval/success_mean"] = float(np.mean(all_successes))

    all_returns: list[float] = []
    for task_name, returns in task_returns.items():
        safe_name = task_name.replace("-", "_").replace(" ", "_")
        metrics[f"eval/return/{safe_name}"] = float(np.mean(returns))
        all_returns.extend(returns)
    if all_returns:
        metrics["eval/return_mean"] = float(np.mean(all_returns))

    metrics["eval/episodes_completed"] = float(completed.sum())
    metrics["eval/task_count"] = float(len(task_success))
    if task_id_rows:
        metrics["eval/task_id_valid_fraction"] = task_id_valid_rows / task_id_rows
        metrics["eval/task_id_unique_count"] = float(len(task_ids_seen))

    if all_actions:
        actions = np.concatenate(all_actions, axis=0)
        xyz = actions[:, : min(3, actions.shape[-1])]
        metrics["eval/action_xyz_near_bound_fraction"] = float(np.mean(np.abs(xyz) >= 0.95))
        for index in range(actions.shape[-1]):
            label = _ACTION_LABELS[index] if index < len(_ACTION_LABELS) else str(index)
            component = actions[:, index]
            metrics[f"eval/action_{label}_mean"] = float(component.mean())
            metrics[f"eval/action_{label}_std"] = float(component.std())
            metrics[f"eval/action_{label}_frac_saturated"] = float(np.mean(np.abs(component) >= 1.0 - 1e-6))

    if all_pre_means and all_pre_stds:
        pre_means = np.concatenate(all_pre_means, axis=0)
        pre_stds = np.concatenate(all_pre_stds, axis=0)
        for index in range(pre_means.shape[-1]):
            label = _ACTION_LABELS[index] if index < len(_ACTION_LABELS) else str(index)
            metrics[f"eval/action_{label}_pre_mean"] = float(pre_means[:, index].mean())
            metrics[f"eval/action_{label}_pre_std"] = float(pre_stds[:, index].mean())

    if all_task_sensitivity:
        metrics["eval/action_task_id_sensitivity"] = float(np.concatenate(all_task_sensitivity).mean())
    if all_proprio_sensitivity:
        metrics["eval/action_proprio_sensitivity"] = float(np.concatenate(all_proprio_sensitivity).mean())
    if all_visual_sensitivity:
        metrics["eval/action_visual_sensitivity"] = float(np.concatenate(all_visual_sensitivity).mean())
    if all_success_probability:
        metrics["eval/predicted_success_probability"] = float(np.concatenate(all_success_probability).mean())
    if all_constraint_probability:
        constraint_probability_values = np.concatenate(all_constraint_probability, axis=0)
        constraint_target_values = np.concatenate(all_constraint_target, axis=0)
        metrics["eval/pred_constraint_brier"] = float(
            np.mean((constraint_probability_values - constraint_target_values) ** 2)
        )
        for index, label in enumerate(("workspace", "lag", "joint_limit")):
            probability = constraint_probability_values[:, index]
            target = constraint_target_values[:, index] >= 0.5
            predicted = probability >= 0.5
            tp = float(np.sum(predicted & target))
            fp = float(np.sum(predicted & ~target))
            fn = float(np.sum(~predicted & target))
            metrics[f"eval/pred_constraint_{label}_prob"] = float(probability.mean())
            metrics[f"eval/pred_constraint_{label}_rate"] = float(target.mean())
            metrics[f"eval/pred_constraint_{label}_brier"] = float(np.mean((probability - target) ** 2))
            metrics[f"eval/pred_constraint_{label}_precision"] = tp / max(1.0, tp + fp)
            metrics[f"eval/pred_constraint_{label}_recall"] = tp / max(1.0, tp + fn)
    for name, predictions, targets in (
        ("retained", all_retained_prediction, all_retained_target),
        ("achieved", all_achieved_prediction, all_achieved_target),
    ):
        if predictions:
            prediction_values = np.concatenate(predictions, axis=0)
            target_values = np.concatenate(targets, axis=0)
            error = prediction_values - target_values
            metrics[f"eval/pred_constraint_{name}_mae_m"] = float(np.abs(error).mean())
            metrics[f"eval/pred_constraint_{name}_vector_error_m"] = float(np.linalg.norm(error, axis=-1).mean())
            for index, axis in enumerate("xyz"):
                metrics[f"eval/pred_constraint_{name}_{axis}_mae_m"] = float(np.abs(error[:, index]).mean())

    # Aggregate policy conditioning by pinned task slot. A healthy MT actor may
    # legitimately share coarse reaching motion, but these series reveal when
    # one task (or every task) collapses to the same state-independent command.
    per_task_actions: dict[str, list[np.ndarray]] = {}
    per_task_conditioning: dict[str, dict[str, list[float]]] = {}
    for slot, task_name in enumerate(slot_task_names):
        safe_name = (task_name or f"env_{slot}").replace("-", "_").replace(" ", "_")
        per_task_actions.setdefault(safe_name, []).extend(slot_all_actions[slot])
        conditioning = per_task_conditioning.setdefault(
            safe_name,
            {"task_id": [], "proprio": [], "visual": []},
        )
        conditioning["task_id"].extend(slot_task_sensitivity[slot])
        conditioning["proprio"].extend(slot_proprio_sensitivity[slot])
        conditioning["visual"].extend(slot_visual_sensitivity[slot])

    for safe_name, task_actions in per_task_actions.items():
        if task_actions:
            actions = np.asarray(task_actions)
            for index in range(actions.shape[-1]):
                label = _ACTION_LABELS[index] if index < len(_ACTION_LABELS) else str(index)
                metrics[f"eval/action_{label}_mean/{safe_name}"] = float(actions[:, index].mean())
                metrics[f"eval/action_{label}_std/{safe_name}"] = float(actions[:, index].std())
        for source, values in per_task_conditioning[safe_name].items():
            if values:
                metrics[f"eval/action_{source}_sensitivity/{safe_name}"] = float(np.mean(values))

    trace_columns = ["task", "timestep"]
    trace_columns += [f"action_{label}" for label in _ACTION_LABELS]
    trace_columns += [f"pre_mean_{label}" for label in _ACTION_LABELS]
    trace_columns += [f"pre_std_{label}" for label in _ACTION_LABELS]
    trace_rows: list[list[str | int | float]] = []
    traced_tasks: set[str] = set()
    for slot, task_name in enumerate(slot_task_names):
        if task_name is None:
            task_name = f"env_{slot}"
        if task_name in traced_tasks:
            continue
        traced_tasks.add(task_name)
        trace = np.asarray(slot_actions[slot])
        for step, action in enumerate(trace):
            pre_mean = slot_pre_means[slot][step] if step < len(slot_pre_means[slot]) else np.full(len(action), np.nan)
            pre_std = slot_pre_stds[slot][step] if step < len(slot_pre_stds[slot]) else np.full(len(action), np.nan)
            trace_rows.append(
                [task_name, step]
                + [float(value) for value in action]
                + [float(value) for value in pre_mean]
                + [float(value) for value in pre_std]
            )

    metrics.update(_actor_parameter_metrics(agent))

    for diag_name, diag_vals in ctrl_diags.items():
        metrics[f"eval/ctrl_{diag_name}"] = float(np.mean(diag_vals))
    per_task_ctrl_metrics = {
        "frac_stuck",
        "frac_undertracking",
        "path_ratio_mean",
        "frac_ws_clamped",
        "frac_lag_clamped",
        "frac_joint_limit_clamped",
    }
    for task_name, task_diags in ctrl_diags_by_task.items():
        safe_name = task_name.replace("-", "_").replace(" ", "_")
        for diag_name, diag_vals in task_diags.items():
            if diag_name in per_task_ctrl_metrics:
                metrics[f"eval/ctrl_{diag_name}/{safe_name}"] = float(np.mean(diag_vals))
    for diag_name, diag_vals in reward_diags.items():
        metrics[f"eval/reward_{diag_name}"] = float(np.mean(diag_vals))

    action_trace = ActionTrace(columns=trace_columns, rows=trace_rows) if capture_artifacts and trace_rows else None
    return EvalResult(
        metrics=metrics,
        video=np.stack(video_frames) if video_frames else None,
        action_trace=action_trace,
    )


def _actor_parameter_metrics(agent: Any) -> dict[str, float]:
    """Prove which actor evaluation used and whether its parameters changed."""
    ac = getattr(agent, "ac", None)
    actor = getattr(ac, "actor", None)
    if actor is None:
        return {}

    with torch.no_grad():
        params = list(actor.parameters())
        square_sum = sum(
            (p.detach().double().square().sum() for p in params),
            start=torch.zeros((), device=params[0].device, dtype=torch.float64),
        )
        # A deterministic, order-sensitive-enough scalar checksum for charts.
        # It is not a cryptographic integrity check; paired with the norm it
        # makes a stale actor immediately visible between eval milestones.
        checksum = sum(
            (
                (index + 1) * (p.detach().double().sum() + 0.5 * p.detach().double().abs().sum())
                for index, p in enumerate(params)
            ),
            start=torch.zeros((), device=params[0].device, dtype=torch.float64),
        )
        metrics = {
            "eval/actor_param_norm": float(torch.sqrt(square_sum).cpu()),
            "eval/actor_param_checksum": float(checksum.cpu()),
            "eval/actor_param_count": float(sum(p.numel() for p in params)),
            "eval/agent_training_mode": float(bool(getattr(agent, "training", False))),
        }

        frozen = getattr(ac, "_frozen_actor", None)
        if frozen is not None:
            frozen_params = list(frozen.parameters())
            if len(frozen_params) == len(params):
                max_diff = max(
                    (p.detach() - fp.detach()).abs().max() for p, fp in zip(params, frozen_params, strict=True)
                )
                shared = sum(p.data_ptr() == fp.data_ptr() for p, fp in zip(params, frozen_params, strict=True))
                metrics["eval/actor_live_frozen_max_diff"] = float(max_diff.cpu())
                metrics["eval/actor_live_frozen_shared_fraction"] = shared / len(params)
        return metrics


def _run(cfg: DictConfig) -> EvalResult:
    """Evaluate a saved checkpoint.

    Composition root for ``configs/inference/evaluate.yaml``: builds envs and
    the agent, restores ``cfg.checkpoint``, and reports the same metrics the
    training loop logs under ``eval/``.
    """
    from dreamer_arm.core.model import Dreamer
    from dreamer_arm.envs.sim.factory import build_from_config
    from dreamer_arm.utils.seed import set_seed_everywhere

    if cfg.checkpoint is None:
        raise ValueError("inference requires a checkpoint: pass checkpoint=<path/to/best.pt>")

    set_seed_everywhere(int(cfg.seed))
    envs = build_from_config(cfg, viewer=bool(cfg.envs.sim.get("viewer", False)))
    try:
        agent = Dreamer(cfg.core.model, envs.observation_space, envs.action_space).to(cfg.device)
        ckpt = torch.load(str(cfg.checkpoint), map_location=cfg.device, weights_only=False)
        agent.load_checkpoint_state(ckpt["agent"])
        log.info("loaded %s (trained to step %d)", cfg.checkpoint, int(ckpt["step"]))

        with phase("eval"):
            result = evaluate(agent, envs, int(cfg.eval.episodes))
    finally:
        envs.close()

    for name, value in sorted(result.metrics.items()):
        log.info("  %-40s %.4f", name, value)
    return result


def run() -> object:
    """``python -m dreamer_arm.inference.evaluate`` (also ``pixi run eval``)."""
    from dreamer_arm.utils.config import dispatch, run_hydra

    return run_hydra(dispatch, config_name="inference/evaluate", selector=("inference", "inference"))


if __name__ == "__main__":
    run()
