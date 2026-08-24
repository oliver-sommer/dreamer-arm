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

    push      YAM-specific staged push diagnostic.  Moves a smooth Cartesian
              reference above/behind the puck, descends, settles, then pushes
              through the puck.  Unlike SawyerPushV3Policy it has no hard
              threshold that can chatter between approach and descend.

``sweep=true`` runs the mode's evaluation (``probe`` or ``iid``) across a
small grid of controller speed, damping, joint-speed, nullspace, orientation,
and lookahead settings, printing one summary row per field/value combination.
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
    "max_ee_speed_m_s": (0.15, 0.25, 0.40),
    "damping": (0.05, 0.10, 0.15),
    "max_joint_speed_rad_s": (1.0, 2.0, 3.0),
    "nullspace_gain": (0.5, 1.0),
    "ori_weight": (0.0, 0.1, 0.3, 1.0),
    "joint_target_horizon_s": (0.05, 0.10, 0.20),
}

_SERVO_CHECKPOINTS: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class DiagnosticResult:
    """Numeric output returned by a controller diagnostic mode."""

    mode: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class PushStageResult:
    """Pose/constraint trace summary for one smooth-push stage."""

    steps: int
    success: bool
    tcp_error_m: float
    tcp_z_min_m: float
    tcp_z_max_m: float
    z_reversals: int
    ori_error_max_rad: float
    joint_limit_clamp_fraction: float


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
        dq_clamped = joint_limit_clamped = ori_capped = n = 0
        for _t in range(horizon):
            inner.step(action)
            diag = arm.last_diagnostics
            tcp = _tcp(inner, gid)
            if diag:
                cmd = float(diag.get("cmd_norm", 0.0))
                if cmd > 1e-4:
                    achieved = tcp - prev_tcp
                    achieved_norm = float(np.linalg.norm(achieved))
                    desired = np.array([diag.get(f"cmd_{axis}", 0.0) for axis in "xyz"])
                    desired_norm = float(np.linalg.norm(desired))
                    ratios.append(float(achieved @ desired) / (desired_norm * cmd) if desired_norm > 1e-12 else 0.0)
                    if achieved_norm > 1e-9:
                        cosines.append(float(achieved @ direction) / achieved_norm)
                dq_clamped += int(diag.get("dq_clamped", 0.0) > 0.0)
                joint_limit_clamped += int(diag.get("joint_limit_clamped", 0.0) > 0.0)
                ori_capped += int(diag.get("ori_capped", 0.0) > 0.0)
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
                "frac_joint_limit_clamped": joint_limit_clamped / n if n else 0.0,
                "frac_ori_capped": ori_capped / n if n else 0.0,
            }
        )
    inner.close()

    agg_keys = (
        "track_mean",
        "track_tail_mean",
        "cos_mean",
        "frac_stuck",
        "frac_dq_clamped",
        "frac_joint_limit_clamped",
        "frac_ori_capped",
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
        "jnt_clmp",
        "ori_cap",
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
            row["frac_joint_limit_clamped"],
            row["frac_ori_capped"],
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
        agg["frac_joint_limit_clamped"],
        agg["frac_ori_capped"],
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
    diag_counts = {
        "dq_clamped": 0,
        "joint_limit_clamped": 0,
        "ori_capped": 0,
    }
    diag_n = 0
    for _ep in range(n_episodes):
        inner.reset()
        prev_tcp = _tcp(inner, gid)
        for _t in range(horizon):
            a = np.clip(rng.normal(0.0, action_std, 4), -1.0, 1.0).astype(np.float32)
            for _r in range(action_repeat):
                inner.step(a)
                diag = arm.last_diagnostics
                tcp = _tcp(inner, gid)
                if diag:
                    diag_n += 1
                    for key in diag_counts:
                        diag_counts[key] += int(diag.get(key, 0.0) > 0.0)
                cmd = float(diag.get("cmd_norm", 0.0)) if diag else 0.0
                if cmd > 1e-4:
                    achieved = tcp - prev_tcp
                    desired = np.array([diag.get(f"cmd_{axis}", 0.0) for axis in "xyz"], dtype=np.float64)
                    desired_norm = float(np.linalg.norm(desired))
                    ratios.append(float(achieved @ desired) / (desired_norm * cmd) if desired_norm > 1e-12 else 0.0)
                prev_tcp = tcp
    inner.close()
    arr = np.array(ratios) if ratios else np.zeros(1)
    result = {
        "track_ratio_mean": float(arr.mean()),
        "frac_stuck": float((arr < 0.25).mean()),
        "n_samples": float(len(ratios)),
    }
    result.update({f"frac_{key}": count / diag_n if diag_n else 0.0 for key, count in diag_counts.items()})
    return result


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


def _advance_push_reference(
    inner: Any,
    arm: Any,
    gid: int,
    target: np.ndarray,
    *,
    budget: int,
    reference_step_m: float,
    viewer: Any | None = None,
    realtime: bool = False,
) -> PushStageResult:
    """Pursue one push waypoint with smooth Cartesian velocity commands."""
    import time

    settled = 0
    success = False
    error = float("inf")
    tcp_z_min = float("inf")
    tcp_z_max = -float("inf")
    previous_z: float | None = None
    previous_dz = 0.0
    z_reversals = 0
    ori_error_max = 0.0
    joint_limit_clamps = 0
    for _step in range(1, budget + 1):
        delta = target - _tcp(inner, gid)
        distance = float(np.linalg.norm(delta))
        requested_step = np.zeros(3)
        if distance > 0.0:
            requested_step = delta * min(1.0, reference_step_m / distance)

        # This is intentionally a controller/mechanics diagnostic, not an
        # agent policy.  It retains the complete IK, joint-servo, collision,
        # and reward paths under test.
        dt = float(inner.model.opt.timestep) * int(inner.frame_skip)
        xyz = requested_step / (dt * arm._cfg.max_ee_speed_m_s)
        _obs, _reward, _terminated, _truncated, info = inner.step(np.array([*xyz, 1.0], dtype=np.float32))
        success = success or float(info.get("success", 0.0)) >= 1.0
        tcp = _tcp(inner, gid)
        error = float(np.linalg.norm(tcp - target))
        settled = settled + 1 if distance <= reference_step_m and error < 0.012 else 0

        tcp_z = float(tcp[2])
        tcp_z_min = min(tcp_z_min, tcp_z)
        tcp_z_max = max(tcp_z_max, tcp_z)
        if previous_z is not None:
            dz = tcp_z - previous_z
            # Ignore sub-0.1 mm solver/contact noise when counting a visible
            # reversal; raw sign changes greatly overstate the wiggle.
            if abs(dz) >= 1e-4:
                if previous_dz and dz * previous_dz < 0.0:
                    z_reversals += 1
                previous_dz = dz
        previous_z = tcp_z

        diagnostics = arm.last_diagnostics or {}
        ori_error_max = max(ori_error_max, float(diagnostics.get("ori_error_norm", 0.0)))
        joint_limit_clamps += int(diagnostics.get("joint_limit_clamped", 0.0) > 0.0)

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            if realtime:
                time.sleep(float(inner.model.opt.timestep) * int(inner.frame_skip))

        if settled >= 8 or success:
            break
    return PushStageResult(
        steps=_step,
        success=success,
        tcp_error_m=error,
        tcp_z_min_m=tcp_z_min,
        tcp_z_max_m=tcp_z_max,
        z_reversals=z_reversals,
        ori_error_max_rad=ori_error_max,
        joint_limit_clamp_fraction=joint_limit_clamps / _step,
    )


def _smooth_push_episode(inner: Any, arm: Any, gid: int, viewer: Any | None = None) -> dict[str, float]:
    """Run one non-chattering push trajectory and return outcome metrics."""
    obs, _info = inner.reset()
    puck_start = np.asarray(obs[4:7], dtype=np.float64).copy()
    goal = np.asarray(obs[-3:], dtype=np.float64).copy()

    push_direction = goal - puck_start
    push_direction[2] = 0.0
    push_direction /= max(float(np.linalg.norm(push_direction)), 1e-9)

    # Place the closed gripper just behind the puck, low enough for its pad
    # collision boxes to overlap the puck, then carry the reference beyond the
    # goal so contact is maintained until Meta-World's 5 cm success radius.
    contact = puck_start - 0.04 * push_direction
    contact[2] = puck_start[2] + 0.015
    above = contact.copy()
    above[2] += 0.10
    through_goal = goal + 0.15 * push_direction
    # A mild downward bias counters the wrist's tendency to rise under
    # horizontal contact load.  The realised TCP remains above the table; the
    # reference is a force-producing servo target, not an expected pose.
    through_goal[2] = puck_start[2] - 0.01

    stages = (
        ("approach", above, 130, 0.00225),
        ("descend", contact, 90, 0.00150),
        ("push", through_goal, 260, 0.00300),
    )
    total_steps = 0
    success = False
    for name, target, budget, reference_step_m in stages:
        result = _advance_push_reference(
            inner,
            arm,
            gid,
            target,
            budget=budget,
            reference_step_m=reference_step_m,
            viewer=viewer,
            realtime=viewer is not None,
        )
        total_steps += result.steps
        success = success or result.success
        log.info(
            "  push stage=%-8s steps=%3d tcp_error=%.4f z=[%.4f, %.4f] "
            "z_reversals=%d ori_max=%.3f limit_clamp=%.2f success=%s",
            name,
            result.steps,
            result.tcp_error_m,
            result.tcp_z_min_m,
            result.tcp_z_max_m,
            result.z_reversals,
            result.ori_error_max_rad,
            result.joint_limit_clamp_fraction,
            success,
        )
        if success or (viewer is not None and not viewer.is_running()):
            break

    final_obs = inner._get_obs()
    puck_final = np.asarray(final_obs[4:7], dtype=np.float64)
    return {
        "success": float(success),
        "puck_motion_m": float(np.linalg.norm(puck_final - puck_start)),
        "puck_goal_error_m": float(np.linalg.norm(puck_final - goal)),
        "controller_steps": float(total_steps),
    }


def run_push(cfg: DictConfig) -> dict[str, float]:
    """Run the smooth YAM push, optionally repeating it in a live viewer."""
    import time

    arm_cfg = _arm_cfg_from_cfg(cfg)
    arm_name = str(cfg.envs.sim.arms.name)
    if arm_name != "yam":
        raise ValueError("mode=push requires the YAM arm")
    if str(cfg.envs.sim.task) not in {"push", "push-v3"}:
        raise ValueError("mode=push requires envs.sim.task=push")

    inner, arm, gid = _build_env(arm_cfg, arm_name, "push", int(cfg.seed))
    viewer = None
    try:
        if bool(cfg.viewer):
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(inner.model, inner.data)

        episode = 0
        metrics: dict[str, float] = {}
        while viewer is None or viewer.is_running():
            episode += 1
            metrics = _smooth_push_episode(inner, arm, gid, viewer)
            log.info("smooth push episode=%d metrics=%s", episode, metrics)
            if viewer is None:
                break
            if viewer.is_running():
                time.sleep(1.0)
        return metrics
    finally:
        if viewer is not None:
            viewer.close()
        inner.close()


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


_MODES = {"probe": run_probe, "iid": run_iid, "servo": run_servo, "coverage": run_coverage, "push": run_push}


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
