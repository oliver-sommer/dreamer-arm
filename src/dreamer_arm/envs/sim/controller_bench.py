"""Standalone bench for the configured Cartesian controller.

Builds a single raw (unwrapped) YAM-armed Meta-World env — no rendering, no
Dict observation, no vectorisation — and rolls the DLS-IK controller
(:class:`dreamer_arm.envs.sim.arms.yam.YamArm`) under different action-selection
regimes, printing the per-step diagnostics it already computes
(``env._ctrl_diag``) plus achieved-vs-commanded TCP tracking.  Exists so the
controller's real authority can be measured in seconds, instead of only by
watching ``episode/ctrl_*`` metrics minutes into a training run.

Composition root for ``configs/envs/sim/controller_bench.yaml``. Modes
(``mode=``):

    probe     (default) hold each of +-x/+-y/+-z and ``n_dirs`` random unit
              directions for ``horizon`` control steps from reset; prints a
              per-direction table of tracking ratio (achieved/commanded TCP
              displacement), direction cosine fidelity, and clamp fractions.

    iid       resample a Gaussian action every ``envs.sim.action_repeat`` control
              steps (mirrors ``SyncVectorEnv``'s action-repeat semantics)
              across ``n_episodes`` episodes.  This is the harshest, most
              realistic proxy for early-training ``frac_stuck`` — it
              reproduces the metric end to end without launching training.

    servo     bypasses the IK entirely: holds one joint's ctrl target a fixed
              offset ahead and prints the settling curve (fraction of the
              commanded step realised after 1/2/5/10/20/50/100 control
              steps) — isolates the MuJoCo position-servo dynamics from the
              IK solve.

    coverage  servos the TCP to the corners + centre of each task's goal
              space and object-spawn range; reports final tracking error and
              conditioning at the final approach. Replaces the legacy
              workspace-coverage sweep.

``sweep=true`` runs the mode's evaluation (``probe`` or ``iid``) across a
small grid of ``ee_step_m`` / ``damping`` / ``max_joint_step`` /
``nullspace_gain`` / ``max_lead_m`` instead of the single ``envs.sim.arms.*``
configuration, printing one summary row per (field, value) combination.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from dreamer_arm.utils.logging import phase

log = logging.getLogger(__name__)

_AXIS_DIRS: tuple[np.ndarray, ...] = tuple(
    np.array(v, dtype=np.float64) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
)

# Grid probed by `sweep=true`, one field varied at a time from the base
# (`envs.sim.arms.*`) config.
_SWEEP_GRID: dict[str, tuple[float, ...]] = {
    "ee_step_m": (0.02, 0.05, 0.08),
    "damping": (0.05, 0.10, 0.15),
    "max_joint_step": (0.5, 1.0, 1.5),
    "nullspace_gain": (0.5, 1.0),
    "max_lead_m": (0.10, 0.25, 0.40),
}

_SERVO_CHECKPOINTS: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class DiagnosticResult:
    """Numeric output returned by a controller diagnostic mode."""

    mode: str
    metrics: dict[str, float]


def _build_env(arm_cfg: Any, arm_name: str, task: str, seed: int) -> tuple[Any, Any, int]:
    """Build a raw Meta-World env with the configured arm.

    Mirrors ``tests/envs/sim/arms/test_yam.py::_make_yam_inner``, extended to accept
    a caller-supplied ``ArmConfig`` so probes / sweeps can vary IK tuning.
    """
    import metaworld
    import mujoco

    from dreamer_arm.envs.sim.arms import make_arm

    metaworld.set_active_arm(arm_name)
    task_tag = task if task.endswith("-v3") else f"{task}-v3"
    mt1 = metaworld.MT1(task_tag, seed=seed)
    env_cls = next(iter(mt1.train_classes.values()))
    inner = env_cls(render_mode=None)
    arm = make_arm(arm_name, arm_cfg)
    arm.attach(inner)  # captures home pose (pre-set_task), installs hooks
    inner.set_task(mt1.train_tasks[0])
    gid = int(mujoco.mj_name2id(inner.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"))
    return inner, arm, gid


def _arm_cfg_from_cfg(cfg: DictConfig) -> Any:
    from dreamer_arm.envs.sim.arms import ArmConfig

    return ArmConfig(**OmegaConf.to_container(cfg.envs.sim.arms, resolve=True))


def _tcp(inner: Any, gid: int) -> np.ndarray:
    # .copy() is load-bearing: site_xpos is a persistent MuJoCo buffer
    # mj_step overwrites in place; ControllerMetrics relies on the same rule.
    return np.asarray(inner.data.site_xpos[gid], dtype=np.float64).copy()


def _roll_probe(arm_cfg: Any, arm_name: str, task: str, seed: int, horizon: int, n_dirs: int) -> dict[str, Any]:
    """Hold each axis + ``n_dirs`` random unit directions for ``horizon`` steps."""
    inner, arm, gid = _build_env(arm_cfg, arm_name, task, seed)
    rng = np.random.default_rng(seed)
    directions = list(_AXIS_DIRS)
    for _ in range(n_dirs):
        d = rng.uniform(-1.0, 1.0, 3)
        directions.append(d / max(float(np.linalg.norm(d)), 1e-6))

    rows: list[dict[str, Any]] = []
    for direction in directions:
        inner.reset()
        prev_tcp = _tcp(inner, gid)
        action = np.array([*(0.7 * direction), -1.0], dtype=np.float32)
        ratios: list[float] = []
        cosines: list[float] = []
        dq_clamped = ori_capped = lead_clamped = n = 0
        for _t in range(horizon):
            inner.step(action)
            diag = arm.last_diagnostics
            tcp = _tcp(inner, gid)
            if diag:
                cmd = float(diag.get("cmd_norm", 0.0))
                if cmd > 1e-4:
                    achieved = tcp - prev_tcp
                    achieved_norm = float(np.linalg.norm(achieved))
                    ratios.append(achieved_norm / cmd)
                    if achieved_norm > 1e-9:
                        cosines.append(float(achieved @ direction) / achieved_norm)
                dq_clamped += int(diag.get("dq_clamped", 0.0) > 0.0)
                ori_capped += int(diag.get("ori_capped", 0.0) > 0.0)
                lead_clamped += int(diag.get("lead_clamped", 0.0) > 0.0)
                n += 1
            prev_tcp = tcp
        # "Steady state" = the last quarter of the horizon, once the servo
        # has had time to catch up to the leashed setpoint.
        tail = max(1, len(ratios) // 4)
        rows.append(
            {
                "direction": direction,
                "track_mean": float(np.mean(ratios)) if ratios else 0.0,
                "track_tail_mean": float(np.mean(ratios[-tail:])) if ratios else 0.0,
                "cos_mean": float(np.mean(cosines)) if cosines else 0.0,
                "frac_stuck": float(np.mean([r < 0.25 for r in ratios])) if ratios else 1.0,
                "frac_dq_clamped": dq_clamped / n if n else 0.0,
                "frac_ori_capped": ori_capped / n if n else 0.0,
                "frac_lead_clamped": lead_clamped / n if n else 0.0,
            }
        )
    inner.close()

    agg_keys = (
        "track_mean",
        "track_tail_mean",
        "cos_mean",
        "frac_stuck",
        "frac_dq_clamped",
        "frac_ori_capped",
        "frac_lead_clamped",
    )
    agg = {k: float(np.mean([row[k] for row in rows])) for k in agg_keys}
    return {"rows": rows, "agg": agg}


def run_probe(cfg: DictConfig) -> dict[str, float]:
    arm_cfg = _arm_cfg_from_cfg(cfg)
    result = _roll_probe(
        arm_cfg,
        str(cfg.envs.sim.arms.name),
        str(cfg.envs.sim.task),
        int(cfg.seed),
        int(cfg.horizon),
        int(cfg.n_dirs),
    )
    log.info("probe: task=%s horizon=%d arm_cfg=%s", cfg.envs.sim.task, cfg.horizon, arm_cfg)
    log.info(
        "  %-19s %10s %10s %8s %8s %8s %8s %8s",
        "direction",
        "track",
        "track_tl",
        "cos",
        "stuck",
        "dq_clmp",
        "ori_cap",
        "lead_clm",
    )
    for row in result["rows"]:
        d = row["direction"]
        log.info(
            "  [%+.2f %+.2f %+.2f]  %10.4f %10.4f %8.4f %8.2f %8.2f %8.2f %8.2f",
            d[0],
            d[1],
            d[2],
            row["track_mean"],
            row["track_tail_mean"],
            row["cos_mean"],
            row["frac_stuck"],
            row["frac_dq_clamped"],
            row["frac_ori_capped"],
            row["frac_lead_clamped"],
        )
    agg = result["agg"]
    log.info(
        "  %-19s %10.4f %10.4f %8.4f %8.2f %8.2f %8.2f %8.2f",
        "MEAN",
        agg["track_mean"],
        agg["track_tail_mean"],
        agg["cos_mean"],
        agg["frac_stuck"],
        agg["frac_dq_clamped"],
        agg["frac_ori_capped"],
        agg["frac_lead_clamped"],
    )
    return agg


def _roll_iid(
    arm_cfg: Any,
    arm_name: str,
    task: str,
    seed: int,
    horizon: int,
    n_episodes: int,
    action_std: float,
    action_repeat: int,
) -> dict[str, float]:
    """Resample a Gaussian action every ``action_repeat`` control steps.

    Mirrors ``SyncVectorEnv``'s action-repeat semantics (the same sampled
    action is held for ``action_repeat`` inner ``env.step`` calls), so this
    is the closest single-env proxy for what a training run actually sees.
    """
    inner, arm, gid = _build_env(arm_cfg, arm_name, task, seed)
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    for _ep in range(n_episodes):
        inner.reset()
        prev_tcp = _tcp(inner, gid)
        for _t in range(horizon):
            a = np.clip(rng.normal(0.0, action_std, 4), -1.0, 1.0).astype(np.float32)
            for _r in range(action_repeat):
                inner.step(a)
                diag = arm.last_diagnostics
                tcp = _tcp(inner, gid)
                cmd = float(diag.get("cmd_norm", 0.0)) if diag else 0.0
                if cmd > 1e-4:
                    ratios.append(float(np.linalg.norm(tcp - prev_tcp)) / cmd)
                prev_tcp = tcp
    inner.close()
    arr = np.array(ratios) if ratios else np.zeros(1)
    return {
        "track_ratio_mean": float(arr.mean()),
        "frac_stuck": float((arr < 0.25).mean()),
        "n_samples": float(len(ratios)),
    }


def run_iid(cfg: DictConfig) -> dict[str, float]:
    arm_cfg = _arm_cfg_from_cfg(cfg)
    stats = _roll_iid(
        arm_cfg,
        str(cfg.envs.sim.arms.name),
        str(cfg.envs.sim.task),
        int(cfg.seed),
        int(cfg.horizon),
        int(cfg.n_episodes),
        float(cfg.action_std),
        int(cfg.envs.sim.action_repeat),
    )
    log.info(
        "iid: task=%s n_episodes=%d horizon=%d action_std=%.2f action_repeat=%d arm_cfg=%s",
        cfg.envs.sim.task,
        cfg.n_episodes,
        cfg.horizon,
        cfg.action_std,
        cfg.envs.sim.action_repeat,
        arm_cfg,
    )
    for k, v in sorted(stats.items()):
        log.info("  %-24s %.4f", k, v)
    return stats


def run_servo(cfg: DictConfig) -> dict[str, float]:
    """Bypass the IK: hold one joint's ctrl target ahead, print the settling curve."""
    import mujoco

    arm_cfg = _arm_cfg_from_cfg(cfg)
    arm_name = str(cfg.envs.sim.arms.name)
    if arm_name != "yam":
        raise ValueError("mode=servo requires the YAM position-servo arm")
    inner, arm, _gid = _build_env(arm_cfg, arm_name, str(cfg.envs.sim.task), int(cfg.seed))
    servo = arm.servo_state
    if servo is None:
        raise ValueError("configured arm does not expose position-servo state")
    m, d = inner.model, inner.data
    joint = int(cfg.servo_joint)
    target_delta = float(cfg.servo_target_rad)

    log.info("servo: joint=%d target_delta=%.3f rad frame_skip=%d", joint, target_delta, inner.frame_skip)
    results: dict[str, float] = {}
    for n in _SERVO_CHECKPOINTS:
        inner.reset()
        mujoco.mj_forward(m, d)
        q0 = float(d.qpos[servo.qpos_indices][joint])
        ctrl = d.ctrl.copy()
        ctrl[servo.actuator_ids] = servo.home_qpos
        ctrl[servo.actuator_ids[joint]] = q0 + target_delta
        ctrl[servo.gripper_actuator_id] = 0.0
        inner.do_simulation(ctrl, inner.frame_skip * n)
        realised = float(d.qpos[servo.qpos_indices][joint]) - q0
        frac = realised / target_delta if target_delta else 0.0
        results[f"control_steps={n}"] = frac
        log.info("  control_steps=%3d (%4d physics)  fraction_realised=%.4f", n, inner.frame_skip * n, frac)
    inner.close()
    return results


def _box_targets(low: np.ndarray, high: np.ndarray) -> list[np.ndarray]:
    """Corners + centre of an axis-aligned box, deduplicated."""
    corners = [np.array(c) for c in itertools.product(*zip(low, high, strict=True))]
    corners.append((low + high) / 2.0)
    uniq: list[np.ndarray] = []
    for c in corners:
        if not any(np.allclose(c, u) for u in uniq):
            uniq.append(c)
    return uniq


def _task_targets(inner: Any) -> list[np.ndarray]:
    """TCP probe targets for one task: goal space + object spawn range."""
    targets = _box_targets(
        np.asarray(inner.goal_space.low, dtype=np.float64),
        np.asarray(inner.goal_space.high, dtype=np.float64),
    )
    rrs = inner._random_reset_space
    targets += _box_targets(
        np.asarray(rrs.low[:3], dtype=np.float64),
        np.asarray(rrs.high[:3], dtype=np.float64),
    )
    # Clamp probe heights to a sane band: at least 1cm above the table.
    return [np.array([t[0], t[1], max(t[2], 0.01)]) for t in targets]


def run_coverage(cfg: DictConfig) -> dict[str, float]:
    """Servo the TCP to each task's goal/spawn-range corners; report coverage.

    The P-gain (``/0.01``) and ``servo_steps``/``err_tol`` budget were tuned
    against the old per-step
    relative-delta controller, not the current integrated-setpoint one — a
    corner target near the edge of the arm's reach can still show 0/N covered
    at these settings even though the controller itself tracks that
    direction fine (see ``mode=probe``).  Treat this mode's numbers as
    workspace-coverage signal, and re-tune the gain/budget before trusting an
    exact covered count.
    """
    arm_cfg = _arm_cfg_from_cfg(cfg)
    arm_name = str(cfg.envs.sim.arms.name)
    servo_steps, err_tol, sigma_floor = 150, 0.02, 0.10  # sigma_min on the length-scaled Jacobian; see solve_dls

    log.info("coverage: servo_steps=%d err_tol=%.3f sigma_floor=%.3f", servo_steps, err_tol, sigma_floor)
    results: dict[str, float] = {}
    for task in cfg.coverage_tasks:
        try:
            inner, arm, gid = _build_env(arm_cfg, arm_name, str(task), int(cfg.seed))
            covered = total = 0
            for target in _task_targets(inner):
                inner.reset()
                sig_tail: list[float] = []
                for t in range(servo_steps):
                    tcp = _tcp(inner, gid)
                    a = np.clip((target - tcp) / 0.01, -1.0, 1.0)
                    inner.step(np.array([*a, -1.0], dtype=np.float32))
                    diag = arm.last_diagnostics
                    if diag and t >= servo_steps - 30:
                        sig_tail.append(float(diag.get("sigma_min", 0.0)))
                err = float(np.linalg.norm(_tcp(inner, gid) - target))
                sig_min = float(np.min(sig_tail)) if sig_tail else 0.0
                total += 1
                if err < err_tol and (arm_name != "yam" or sig_min > sigma_floor):
                    covered += 1
            inner.close()
            results[f"{task}/covered"] = float(covered)
            results[f"{task}/total"] = float(total)
            log.info("  %-20s covered=%d/%d", task, covered, total)
        except Exception as exc:  # noqa: BLE001 -- one task's failure must not abort the sweep
            log.warning("  %-20s ERROR %s: %s", task, type(exc).__name__, exc)
            results[f"{task}/covered"] = 0.0
            results[f"{task}/total"] = 0.0
    return results


def run_sweep(cfg: DictConfig) -> None:
    base = _arm_cfg_from_cfg(cfg)
    arm_name = str(cfg.envs.sim.arms.name)
    task, seed, horizon = str(cfg.envs.sim.task), int(cfg.seed), int(cfg.horizon)
    use_iid = str(cfg.mode) == "iid"

    log.info("sweep: base=%s eval=%s", base, "iid" if use_iid else "probe")
    log.info("  %-28s %10s %10s %10s", "field=value", "track", "stuck", "cos")
    for field, values in _SWEEP_GRID.items():
        for value in values:
            arm_cfg = replace(base, **{field: value})
            try:
                if use_iid:
                    stats = _roll_iid(
                        arm_cfg,
                        arm_name,
                        task,
                        seed,
                        horizon,
                        int(cfg.n_episodes),
                        float(cfg.action_std),
                        int(cfg.envs.sim.action_repeat),
                    )
                    track, stuck, cos = stats["track_ratio_mean"], stats["frac_stuck"], float("nan")
                else:
                    result = _roll_probe(arm_cfg, arm_name, task, seed, horizon, int(cfg.n_dirs))
                    track, stuck, cos = (
                        result["agg"]["track_mean"],
                        result["agg"]["frac_stuck"],
                        result["agg"]["cos_mean"],
                    )
            except Exception as exc:  # noqa: BLE001 -- one config's failure must not abort the sweep
                log.warning("  %s=%-20s ERROR %s: %s", field, value, type(exc).__name__, exc)
                continue
            log.info("  %s=%-20s %10.4f %10.4f %10.4f", field, value, track, stuck, cos)


_MODES = {"probe": run_probe, "iid": run_iid, "servo": run_servo, "coverage": run_coverage}


def _run(cfg: DictConfig) -> DiagnosticResult | None:
    """Composition root for ``configs/envs/sim/controller_bench.yaml``."""
    with phase("diag"):
        if bool(cfg.sweep):
            run_sweep(cfg)
            return None
        mode = str(cfg.mode)
        if mode not in _MODES:
            raise ValueError(f"Unknown controller-bench mode: {mode!r}. Expected one of {sorted(_MODES)}.")
        return DiagnosticResult(mode=mode, metrics=_MODES[mode](cfg))


def run() -> object:
    """Run the controller bench through Hydra configuration."""
    from dreamer_arm.utils.config import dispatch, run_hydra

    return run_hydra(dispatch, config_name="envs/sim/controller_bench")


if __name__ == "__main__":
    run()
