"""YAM arm control seam — DLS-IK actuation via Meta-World hooks.

The YAM arm has 6 position-actuated joints (``joint1``…``joint6``) plus a
``gripper`` actuator driving the ``left_finger`` slide joint (``right_finger``
is mirrored via an equality constraint).

Key design choices that address the old failure modes:
- An *integrated* Cartesian setpoint (``_x_des``), leashed to within
  ``max_lead_m`` of the actual TCP, replaces a per-step relative delta.  The
  MuJoCo position servos take several control steps (~10 at the default
  gains) to realise a commanded joint increment; re-deriving the setpoint
  from the freshly measured joint angle every step (as the old design did)
  made it retreat by the same amount each cycle, so the arm crawled at a few
  percent of the commanded rate and never converged (``ctrl_frac_stuck≈1``).
  Integrating the setpoint gives the servo many cycles to catch up instead.
- Constant-λ DLS on a length-scaled Jacobian bounds ``‖dq‖`` everywhere — no
  stuck-detector needed.
- Nullspace posture bias continuously repels from joint limits.
- Joint target clamping (task and posture components independently, not
  velocity gating) keeps the actuator live.
- Position actuators + ``do_simulation`` give real friction for object transport.
- Dual ``[-1, 1]`` clamp: the agent-side clamp (``core/actor_critic.py``,
  ``sanitize_action``) is authoritative for buffer / RSSM; this env-side
  clamp guards the contract boundary against warmup / eval / scripted callers.
"""

from __future__ import annotations

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
        self._x_des: np.ndarray | None = None  # (3,) integrated Cartesian TCP setpoint
        self._ik_cfg: IKConfig | None = None  # built once in attach(); cfg is immutable
        self._jacp: np.ndarray | None = None  # (3, m.nv) Jacobian scratch, reused every step
        self._jacr: np.ndarray | None = None  # (3, m.nv) Jacobian scratch, reused every step
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
        self._ik_cfg = IKConfig(
            ee_step_m=cfg.ee_step_m,
            damping=cfg.damping,
            nullspace_gain=cfg.nullspace_gain,
            ori_gain=cfg.ori_gain,
            joint_margin=cfg.joint_margin,
            max_joint_step=cfg.max_joint_step,
            max_lead_m=cfg.max_lead_m,
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
        assert self._jnt_range is not None
        assert self._ik_cfg is not None
        assert self._jacp is not None
        assert self._jacr is not None

        cfg = self._cfg
        q = d.qpos[self._arm_qadr].copy()
        tcp = np.asarray(d.site_xpos[self._grasp_site_id], dtype=np.float64)

        # Guard: if physics has already blown up (NaN/Inf), don't propagate
        # into ctrl and don't corrupt the integrated setpoint with a NaN TCP
        # — just forward with the current ctrl.
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(tcp)):
            mujoco.mj_forward(m, d)
            return

        # x_des integrates the action across steps rather than resetting to
        # `tcp + delta` every step (see module docstring): the servo gets
        # many control steps to close the gap instead of one that mostly
        # fails.  The leash retracts x_des back to within max_lead_m of the
        # *actual* tcp every step, so it cannot wind up unboundedly ahead
        # while the arm is still catching up.
        if self._x_des is None:
            self._x_des = tcp.copy()
        self._x_des = self._x_des + a[:3] * cfg.ee_step_m
        lead = self._x_des - tcp
        lead_norm = float(np.linalg.norm(lead))
        lead_clamped = lead_norm > cfg.max_lead_m
        if lead_clamped and lead_norm > 0.0:
            lead = lead * (cfg.max_lead_m / lead_norm)
            self._x_des = tcp + lead
        e_pos = lead

        ik_cfg = self._ik_cfg
        e_ori = cfg.ori_gain * quat_log_error(d.site_xmat[self._grasp_site_id], self._quat_home)

        # Cap the orientation term relative to the commanded translation.  The
        # full-quaternion error can reach ‖e_ori‖≈2 (a 180° wrist offset) while
        # the translation error is ‖e_pos‖≤max_lead_m·√3 — uncapped, the 6-D
        # DLS solve spends almost all of dq chasing an orientation the wrist
        # often *cannot* reach (its joints are limited to ±π/2) and starves
        # the position command, dragging the TCP off target and freezing the
        # arm.  Capping orientation to the translation budget (floored at
        # 0.3·ee_step so regulation stays alive for near-zero commands)
        # guarantees position tracking always wins.
        ori_cap = max(float(np.linalg.norm(e_pos)), 0.3 * cfg.ee_step_m)
        ori_norm = float(np.linalg.norm(e_ori))
        ori_capped = ori_norm > ori_cap
        if ori_capped and ori_norm > 0.0:
            e_ori = e_ori * (ori_cap / ori_norm)
        e = np.concatenate([e_pos, e_ori])

        # mj_jacSite overwrites self._jacp/self._jacr in place; J below is a
        # fresh array (np.vstack copies), so reusing the same scratch buffers
        # across steps instead of reallocating them is safe.
        mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self._grasp_site_id)
        J = np.vstack([self._jacp[:, self._arm_dadr], self._jacr[:, self._arm_dadr]])  # (6,6)

        if not np.all(np.isfinite(J)):
            mujoco.mj_forward(m, d)
            return

        # Per-step controller diagnostics (read by MetaWorldEnv into info so the
        # trainer can log episode aggregates: singularity, orientation fighting,
        # joint-velocity saturation, commanded vs achieved TCP motion).
        # `cmd_norm` is the raw per-step action request (‖a[:3]‖·ee_step_m);
        # `err_norm` is the leash-bounded tracking error actually servoed
        # this step — under the old (unintegrated) design the two coincided,
        # but they diverge once the setpoint accumulates across steps.
        diag: dict[str, float] = {
            "cmd_norm": float(np.linalg.norm(a[:3]) * cfg.ee_step_m),
            "err_norm": float(np.linalg.norm(e_pos)),
            "lead_clamped": float(lead_clamped),
            "ori_capped": float(ori_capped),
        }
        self._last_diagnostics = diag
        env._ctrl_diag = diag

        q_target = solve_dls(J, e, q, self._q_home, self._jnt_range, ik_cfg, diag=diag, sigma_min=True)

        # Guard: DLS result should be finite (bounded by joint clamp + λ).
        if not np.all(np.isfinite(q_target)):
            mujoco.mj_forward(m, d)
            return

        g_ctrl = float(_GRIPPER_MAX_OPEN * (1.0 - float(a[3])) / 2.0)

        ctrl = d.ctrl.copy()
        ctrl[self._arm_act_ids] = q_target
        ctrl[self._grip_act_id] = g_ctrl
        env.do_simulation(ctrl, env.frame_skip)

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

        # Re-seed the integrated Cartesian setpoint at the settled TCP so it
        # cannot carry a stale lead across episodes.
        self._x_des = np.asarray(d.site_xpos[self._grasp_site_id], dtype=np.float64).copy()

        # tcp_center = mean of leftEndEffector / rightEndEffector sites
        env.init_tcp = env.tcp_center.copy()
