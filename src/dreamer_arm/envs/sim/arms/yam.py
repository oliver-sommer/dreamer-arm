"""YAM arm control seam — retained Cartesian target and DLS-IK actuation.

The YAM arm has 6 position-actuated joints (``joint1``…``joint6``) plus a
``gripper`` actuator driving the ``left_finger`` slide joint (``right_finger``
is mirrored via an equality constraint).

Control contract:
- XYZ actions are bounded Cartesian velocities.
- Each call rate-integrates velocity into a retained grasp-site position target.
- Workspace and following-error clamps prevent unreachable-target windup.
- Zero velocity holds the commanded pose instead of following disturbances.
- Damped IK, posture bias, and soft joint limits bound the solve.
- Orientation is optional and disabled by default for the restricted wrist.
- The gripper action remains a direct bounded position command.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from dreamer_arm.envs.control.ik import IKConfig, quat_log_error, solve_dls
from dreamer_arm.envs.control.metrics import ServoState
from dreamer_arm.envs.sim.arms.base import ArmConfig

if TYPE_CHECKING:
    pass

_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
_GRASP_SITE_NAME = "grasp_site"
_GRIPPER_ACT_NAME = "gripper"
_GRIPPER_MAX_OPEN = 0.041  # ctrlrange hi = fully open

log = logging.getLogger(__name__)


class YamArm:
    """Arm control seam for the YAM 6-DOF manipulator."""

    def __init__(self, cfg: ArmConfig) -> None:
        self._cfg = cfg
        # Resolved after attach():
        self._arm_act_ids: np.ndarray | None = None  # actuator indices (6,)
        self._grip_act_id: int | None = None  # gripper actuator index
        self._arm_qadr: np.ndarray | None = None  # qpos addresses (6,)
        self._arm_dadr: np.ndarray | None = None  # dof addresses (6,)
        self._jnt_range: np.ndarray | None = None  # (6, 2)
        self._grasp_site_id: int | None = None
        self._q_home: np.ndarray | None = None  # (6,) home joint angles
        self._quat_home: np.ndarray | None = None  # (4,) [w,x,y,z] home EE orientation
        self._p_target: np.ndarray | None = None  # (3,) retained grasp-site position target
        self._ws_low: np.ndarray | None = None  # (3,) Cartesian workspace lower bound
        self._ws_high: np.ndarray | None = None  # (3,) Cartesian workspace upper bound
        self._ik_cfg: IKConfig | None = None  # built once in attach(); cfg is immutable
        self._jacp: np.ndarray | None = None  # (3, m.nv) Jacobian scratch, reused every step
        self._jacr: np.ndarray | None = None  # (3, m.nv) Jacobian scratch, reused every step
        self._velocity = np.zeros(3)  # Cartesian velocity scratch
        self._p_cmd = np.zeros(3)  # unclamped/projected target scratch
        self._target_lag = np.zeros(3)  # target-minus-measurement scratch
        self._last_diagnostics: dict[str, float] | None = None

    @property
    def name(self) -> str:
        return "yam"

    @property
    def last_diagnostics(self) -> Mapping[str, float] | None:
        return self._last_diagnostics

    @property
    def servo_state(self) -> ServoState | None:
        if self._arm_qadr is None or self._arm_act_ids is None or self._q_home is None or self._grip_act_id is None:
            return None
        return ServoState(self._arm_qadr, self._arm_act_ids, self._q_home, self._grip_act_id)

    def attach(self, env: Any) -> None:
        """Resolve model ids, capture home pose, install hooks on *env*.

        Called once after ``env_cls(render_mode=None)`` but before
        ``env.set_task()`` / ``env.reset()``.
        """
        m: Any = env.model
        d: Any = env.data

        self._arm_act_ids = np.array([m.actuator(n).id for n in _ARM_JOINT_NAMES], dtype=np.int32)
        self._grip_act_id = int(m.actuator(_GRIPPER_ACT_NAME).id)

        jids = [m.joint(n).id for n in _ARM_JOINT_NAMES]
        self._arm_qadr = np.array([int(m.jnt_qposadr[j]) for j in jids], dtype=np.int32)
        self._arm_dadr = np.array([int(m.jnt_dofadr[j]) for j in jids], dtype=np.int32)
        self._jnt_range = m.jnt_range[jids].copy()  # (6, 2), radians

        self._grasp_site_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, _GRASP_SITE_NAME))

        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        self._q_home = d.qpos[self._arm_qadr].copy()  # (6,)
        # Home orientation from the grasp-site rotation matrix (gripper-down)
        from dreamer_arm.envs.control.ik import _mat2quat

        self._quat_home = _mat2quat(d.site_xmat[self._grasp_site_id].reshape(3, 3))

        # Cache immutable config and fixed-size scratch arrays across 80 Hz calls.
        cfg = self._cfg
        if not np.isfinite(cfg.max_lag_m) or cfg.max_lag_m <= 0.0:
            raise ValueError(f"max_lag_m must be finite and positive, got {cfg.max_lag_m!r}")
        self._ws_low, self._ws_high, bounds_source = self._resolve_workspace_bounds(env)
        log.info(
            "YAM Cartesian workspace source=%s low=%s high=%s",
            bounds_source,
            self._ws_low,
            self._ws_high,
        )
        self._ik_cfg = IKConfig(
            damping=cfg.damping,
            nullspace_gain=cfg.nullspace_gain * cfg.joint_target_horizon_s,
            ori_weight=cfg.ori_weight,
            joint_margin=cfg.joint_margin,
            max_joint_step=cfg.max_joint_speed_rad_s * cfg.joint_target_horizon_s,
            length_scale=cfg.length_scale,
        )
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

        env._external_actuation = self.actuate
        env._external_reset_hand = self.reset_hand

    def actuate(self, env: Any, action: Any) -> None:
        """Advance physics for one control step using DLS-IK.

        Fully responsible for calling ``do_simulation`` (contract of
        ``_external_actuation`` in sawyer_xyz_env.py:612-626).

        After this returns, ``step()`` calls ``mj_forward`` and reads obs.
        """
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        m: Any = env.model
        d: Any = env.data
        assert self._arm_dadr is not None
        assert self._arm_qadr is not None
        assert self._arm_act_ids is not None
        assert self._grip_act_id is not None
        assert self._grasp_site_id is not None
        assert self._q_home is not None
        assert self._quat_home is not None
        assert self._ws_low is not None
        assert self._ws_high is not None
        assert self._jnt_range is not None
        assert self._ik_cfg is not None
        assert self._jacp is not None
        assert self._jacr is not None

        cfg = self._cfg
        q = d.qpos[self._arm_qadr].copy()
        tcp = np.asarray(d.site_xpos[self._grasp_site_id], dtype=np.float64).copy()

        # Guard: if physics has already blown up (NaN/Inf), do not propagate
        # invalid values into actuator controls.
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(tcp)):
            # A failed command never reached the plant, so discard its target
            # update and resynchronise to a usable measured TCP.  If TCP itself
            # is invalid, retain the last finite target rather than storing NaN.
            self._resync_target(tcp)
            mujoco.mj_forward(m, d)
            return

        if self._p_target is None:
            # reset_hand normally establishes the target.  Lazily initialising
            # here keeps the hook safe if an external caller actuates directly.
            self._p_target = tcp.copy()

        # Interpret policy XYZ as a physical Cartesian velocity and integrate
        # it using the actual 80 Hz control interval.  The workspace projection
        # matches Meta-World's Sawyer mocap bounds; the following-error leash is
        # tracking anti-windup, bounding any transient that contact can store.
        control_dt = float(m.opt.timestep) * int(env.frame_skip)
        np.multiply(a[:3], cfg.max_ee_speed_m_s, out=self._velocity)
        np.multiply(self._velocity, control_dt, out=self._p_cmd)
        self._p_cmd += self._p_target
        ws_clamped = bool(np.any((self._p_cmd < self._ws_low) | (self._p_cmd > self._ws_high)))
        np.clip(self._p_cmd, self._ws_low, self._ws_high, out=self._p_cmd)
        self._target_lag[:] = self._p_cmd
        self._target_lag -= tcp
        lag_clamped = bool(np.any(np.abs(self._target_lag) > cfg.max_lag_m))
        np.clip(self._target_lag, -cfg.max_lag_m, cfg.max_lag_m, out=self._target_lag)
        self._p_target[:] = tcp
        self._p_target += self._target_lag
        e_pos = self._target_lag

        ik_cfg = self._ik_cfg
        ori_error = quat_log_error(d.site_xmat[self._grasp_site_id], self._quat_home)
        angular_velocity = cfg.ori_gain * ori_error if cfg.ori_weight > 0.0 else np.zeros(3)

        # Bound optional orientation feedback independently.  The YAM default
        # sets ori_weight=0 because preserving the home wrist pose across the
        # full task workspace can conflict with translation.
        ori_cap = cfg.max_ori_speed_rad_s
        ori_norm = float(np.linalg.norm(angular_velocity))
        ori_capped = cfg.ori_weight > 0.0 and ori_norm > ori_cap
        if ori_capped and ori_norm > 0.0:
            angular_velocity = angular_velocity * (ori_cap / ori_norm)
        e_ori = angular_velocity * cfg.joint_target_horizon_s
        e = np.concatenate([e_pos, e_ori])

        # mj_jacSite overwrites self._jacp/self._jacr in place; J below is a
        # fresh array (np.vstack copies), so reusing the same scratch buffers
        # across steps instead of reallocating them is safe.
        mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self._grasp_site_id)
        J = np.vstack([self._jacp[:, self._arm_dadr], self._jacr[:, self._arm_dadr]])  # (6,6)

        if not np.all(np.isfinite(J)):
            self._resync_target(tcp)
            mujoco.mj_forward(m, d)
            return

        # Per-step controller diagnostics (read by MetaWorldEnv into info so the
        # trainer can log episode aggregates: singularity, orientation fighting,
        # joint-velocity saturation, commanded vs achieved TCP motion).
        # cmd_* describes the physical displacement requested during this
        # control interval. ik_step_norm remains backward compatible and now
        # equals the retained target lag used as the translational IK task.
        diag: dict[str, float] = {
            "cmd_norm": float(np.linalg.norm(self._velocity) * control_dt),
            "cmd_x": float(self._velocity[0] * control_dt),
            "cmd_y": float(self._velocity[1] * control_dt),
            "cmd_z": float(self._velocity[2] * control_dt),
            "cmd_speed_m_s": float(np.linalg.norm(self._velocity)),
            "ik_step_norm": float(np.linalg.norm(e_pos)),
            "target_lag_norm": float(np.linalg.norm(e_pos)),
            "lag_clamped": float(lag_clamped),
            "ws_clamped": float(ws_clamped),
            "ori_capped": float(ori_capped),
            "ori_error_norm": float(np.linalg.norm(ori_error)),
            "ori_task_norm": float(np.linalg.norm(e_ori)),
        }
        self._last_diagnostics = diag
        env._ctrl_diag = diag

        q_target = solve_dls(J, e, q, self._q_home, self._jnt_range, ik_cfg, diag=diag, sigma_min=True)

        # Guard: DLS result should be finite (bounded by joint clamp + λ).
        if not np.all(np.isfinite(q_target)):
            self._resync_target(tcp)
            diag["target_lag_norm"] = 0.0
            diag["ik_step_norm"] = 0.0
            mujoco.mj_forward(m, d)
            return

        g_ctrl = float(_GRIPPER_MAX_OPEN * (1.0 - float(a[3])) / 2.0)

        ctrl = d.ctrl.copy()
        ctrl[self._arm_act_ids] = q_target
        ctrl[self._grip_act_id] = g_ctrl
        env.do_simulation(ctrl, env.frame_skip)

    def _resolve_workspace_bounds(self, env: Any) -> tuple[np.ndarray, np.ndarray, str]:
        """Prefer Meta-World's task bounds, then config, else unbounded."""

        env_low = getattr(env, "mocap_low", None)
        env_high = getattr(env, "mocap_high", None)
        if env_low is not None and env_high is not None:
            return (*self._validate_workspace_bounds(env_low, env_high, "env.mocap_low/high"), "environment")

        cfg_low = self._cfg.workspace_low
        cfg_high = self._cfg.workspace_high
        if (cfg_low is None) != (cfg_high is None):
            raise ValueError("workspace_low and workspace_high must be supplied together")
        if cfg_low is not None and cfg_high is not None:
            return (*self._validate_workspace_bounds(cfg_low, cfg_high, "ArmConfig"), "config")

        return np.full(3, -np.inf), np.full(3, np.inf), "unbounded"

    @staticmethod
    def _validate_workspace_bounds(low: Any, high: Any, source: str) -> tuple[np.ndarray, np.ndarray]:
        low_arr = np.asarray(low, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        if low_arr.shape != (3,) or high_arr.shape != (3,):
            raise ValueError(f"{source} workspace bounds must both have shape (3,)")
        if not np.all(np.isfinite(low_arr)) or not np.all(np.isfinite(high_arr)):
            raise ValueError(f"{source} workspace bounds must be finite")
        if np.any(low_arr > high_arr):
            raise ValueError(f"{source} workspace lower bound exceeds upper bound")
        return low_arr.copy(), high_arr.copy()

    def _resync_target(self, tcp: np.ndarray) -> None:
        """Roll a failed command back to measured state without storing NaNs."""

        if np.all(np.isfinite(tcp)):
            self._p_target = tcp.copy()

    def reset_hand(self, env: Any, steps: int) -> None:
        """Servo to home pose and set ``env.init_tcp``.

        ``env.init_tcp`` is REQUIRED by Meta-World reward functions
        (e.g., ``_gripper_caging_reward``, line 889 in sawyer_xyz_env.py).
        """
        assert self._arm_act_ids is not None
        assert self._grip_act_id is not None
        assert self._q_home is not None
        assert self._grasp_site_id is not None

        m: Any = env.model
        d: Any = env.data

        n_frames = max(steps, self._cfg.settle_steps)
        ctrl = d.ctrl.copy()
        ctrl[self._arm_act_ids] = self._q_home
        ctrl[self._grip_act_id] = _GRIPPER_MAX_OPEN  # open at reset
        env.do_simulation(ctrl, n_frames)
        mujoco.mj_forward(m, d)

        # The controlled point is grasp_site.  Meta-World's tcp_center below is
        # the mean of the finger sites and is intentionally a different point.
        self._p_target = np.asarray(d.site_xpos[self._grasp_site_id], dtype=np.float64).copy()

        # tcp_center = mean of leftEndEffector / rightEndEffector sites
        env.init_tcp = env.tcp_center.copy()
