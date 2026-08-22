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


@dataclass
class EvalResult:
    """Outcome of one evaluation pass.

    Attributes:
        metrics: Ready-to-log scalars, already namespaced under ``eval/``.
        video:   ``(T, H, W, C)`` frames from env 0's first episode, or
                 ``None`` when the envs expose no ``scene`` observation.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    video: np.ndarray | None = None


def evaluate(agent: Any, envs: Any, episodes: int) -> EvalResult:
    """Run at least ``episodes`` evaluation episodes and summarise them.

    Episodes are collected in parallel across the vector env, so the effective
    count is rounded up to a whole number of rounds across ``envs.num_envs``.

    Args:
        agent:    Dreamer agent with ``get_initial_state``, ``act``, ``device``.
        envs:     ``SyncVectorEnv`` to evaluate in.  Reset with a fixed seed,
                  so a shared train/eval env must be re-reset by the caller
                  afterwards to resume collection.
        episodes: Minimum number of episodes to complete.

    Returns:
        An :class:`EvalResult` with per-task success rates, the mean success
        across all episodes, and any controller diagnostics the envs report.
    """
    n = envs.num_envs
    device = agent.device
    obs_keys = sorted(envs.observation_space.spaces.keys())
    num_rounds = max(1, math.ceil(episodes / n))

    task_success: dict[str, list[float]] = {}
    # EEController diagnostics, averaged across eval episodes (YAM only).
    ctrl_diags: dict[str, list[float]] = {}

    obs_np = envs.reset(seed=EVAL_SEED)
    state = agent.get_initial_state(n)
    is_first = np.ones(n, dtype=bool)
    completed = np.zeros(n, dtype=np.int32)  # episodes done per env slot

    video_frames: list[np.ndarray] = []
    video_done = False

    while completed.min() < num_rounds:
        if not video_done and "scene" in obs_np:
            video_frames.append(obs_np["scene"][0])

        obs_torch: dict[str, torch.Tensor] = {k: torch.from_numpy(obs_np[k]).to(device) for k in obs_keys}
        obs_torch["is_first"] = torch.from_numpy(is_first).to(device)

        with torch.no_grad():
            action_t, next_state = agent.act(obs_torch, state, eval_mode=True)
        action_np = action_t.detach().cpu().numpy()

        obs_next_np, _rewards, terms, truncs, info = envs.step(action_np)
        done = terms | truncs

        for i in range(n):
            if done[i] and completed[i] < num_rounds:
                fin = info["final_info"][i]
                task_name = fin.get("task_name", f"env_{i}") if fin is not None else f"env_{i}"
                success = float(fin.get("success", 0.0)) if fin is not None else 0.0
                task_success.setdefault(task_name, []).append(success)
                if fin is not None:
                    for k, v in fin.get("ctrl_diag", {}).items():
                        ctrl_diags.setdefault(k, []).append(float(v))
                completed[i] += 1

        if not video_done and done[0]:
            video_done = True

        is_first = done.copy()
        state = next_state
        if done.any():
            done_t = torch.from_numpy(done).to(device)
            state["prev_action"][done_t] = 0.0

        obs_np = obs_next_np

    metrics: dict[str, float] = {}
    all_successes: list[float] = []
    for task_name, successes in task_success.items():
        safe_name = task_name.replace("-", "_").replace(" ", "_")
        metrics[f"eval/success/{safe_name}"] = float(np.mean(successes))
        all_successes.extend(successes)

    if all_successes:
        metrics["eval/success_mean"] = float(np.mean(all_successes))

    for diag_name, diag_vals in ctrl_diags.items():
        metrics[f"eval/ctrl_{diag_name}"] = float(np.mean(diag_vals))

    return EvalResult(metrics=metrics, video=np.stack(video_frames) if video_frames else None)


def _run(cfg: DictConfig) -> EvalResult:
    """Evaluate a saved checkpoint.

    Composition root for ``configs/inference/evaluate.yaml``: builds envs and
    the agent, restores ``cfg.checkpoint``, and reports the same metrics the
    training loop logs under ``eval/``.
    """
    from dreamer_arm.core.model import Dreamer
    from dreamer_arm.envs.factory import build_from_config
    from dreamer_arm.utils.seed import set_seed_everywhere

    if cfg.checkpoint is None:
        raise ValueError("inference requires a checkpoint: pass checkpoint=<path/to/checkpoint.pt>")

    set_seed_everywhere(int(cfg.seed))
    envs = build_from_config(cfg, viewer=bool(cfg.envs.get("viewer", False)))
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
