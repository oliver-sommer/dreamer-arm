"""YAM arm control seam — DLS-IK actuation via Meta-World hooks.

The YAM arm has 6 position-actuated joints (``joint1``…``joint6``) plus a
``gripper`` actuator driving the ``left_finger`` slide joint (``right_finger``
is mirrored via an equality constraint).

Key design choices that address the old failure modes:
- Constant-λ DLS bounds ``‖dq‖`` everywhere — no stuck-detector needed.
- Nullspace posture bias continuously repels from joint limits.
- Joint target clamping (not velocity gating) keeps the actuator live.
- Position actuators + ``do_simulation`` give real friction for object transport.
- Dual ``[-1, 1]`` clamp: the agent-side clamp (``dreamer.py:272``) is
  authoritative for buffer / RSSM; this env-side clamp guards the contract
  boundary against warmup / eval / scripted callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from dreamer_arm.envs.arms.base import ArmConfig
from dreamer_arm.envs.control import IKConfig, quat_log_error, solve_dls

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

    @property
    def name(self) -> str:
        return "yam"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(self, env: Any) -> None:
        """Resolve model ids, capture home pose, install hooks on *env*.

        Called once after ``env_cls(render_mode=None)`` but before
        ``env.set_task()`` / ``env.reset()``.
        """
        m: Any = env.model
        d: Any = env.data

        # -- resolve actuator ids (order matches yam_xyz_base_dependencies.xml) --
        self._arm_act_ids = np.array([m.actuator(n).id for n in _ARM_JOINT_NAMES], dtype=np.int32)
        self._grip_act_id = int(m.actuator(_GRIPPER_ACT_NAME).id)

        # -- resolve joint qpos / dof addresses --
        jids = [m.joint(n).id for n in _ARM_JOINT_NAMES]
        self._arm_qadr = np.array([int(m.jnt_qposadr[j]) for j in jids], dtype=np.int32)
        self._arm_dadr = np.array([int(m.jnt_dofadr[j]) for j in jids], dtype=np.int32)
        self._jnt_range = m.jnt_range[jids].copy()  # (6, 2), radians

        # -- grasp site --
        self._grasp_site_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, _GRASP_SITE_NAME))

        # -- capture home configuration from the post-reset state --
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        self._q_home = d.qpos[self._arm_qadr].copy()  # (6,)
        # Home orientation from the grasp-site rotation matrix (gripper-down)
        from dreamer_arm.envs.control import _mat2quat

        self._quat_home = _mat2quat(d.site_xmat[self._grasp_site_id].reshape(3, 3))

        # -- install hooks --
        env._external_actuation = self.actuate
        env._external_reset_hand = self.reset_hand

    # ------------------------------------------------------------------
    # _external_actuation hook
    # ------------------------------------------------------------------

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

        # --- task-space error ---
        cfg = self._cfg
        ik_cfg = IKConfig(
            ee_step_m=cfg.ee_step_m,
            damping=cfg.damping,
            nullspace_gain=cfg.nullspace_gain,
            ori_gain=cfg.ori_gain,
            joint_margin=cfg.joint_margin,
            max_joint_step=cfg.max_joint_step,
        )
        e_pos = a[:3] * cfg.ee_step_m
        e_ori = cfg.ori_gain * quat_log_error(d.site_xmat[self._grasp_site_id], self._quat_home)

        # Cap the orientation term relative to the commanded translation.  The
        # full-quaternion error can reach ‖e_ori‖≈2 (a 180° wrist offset) while
        # the translation command is ‖e_pos‖≤ee_step·√3 (~0.087 at ee_step=0.05)
        # — a ~20x scale mismatch.  Uncapped, the 6-D DLS solve spends almost all
        # of dq chasing an orientation the wrist often *cannot* reach (its joints
        # are limited to ±π/2) and starves the position command, dragging the TCP
        # off target and freezing the arm.  Capping orientation to the
        # translation budget (floored at 0.3·ee_step so regulation stays alive
        # for near-zero commands) guarantees position tracking always wins.
        ori_cap = max(float(np.linalg.norm(e_pos)), 0.3 * cfg.ee_step_m)
        ori_norm = float(np.linalg.norm(e_ori))
        ori_capped = ori_norm > ori_cap
        if ori_capped and ori_norm > 0.0:
            e_ori = e_ori * (ori_cap / ori_norm)
        e = np.concatenate([e_pos, e_ori])

        # --- Jacobian at grasp_site ---
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self._grasp_site_id)
        J = np.vstack([jacp[:, self._arm_dadr], jacr[:, self._arm_dadr]])  # (6,6)

        # --- DLS step ---
        q = d.qpos[self._arm_qadr].copy()

        # Guard: if physics has already blown up (NaN/Inf positions) do not
        # propagate NaN into ctrl — just forward with the current ctrl.
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(J)):
            mujoco.mj_forward(m, d)
            return

        # Per-step controller diagnostics (read by MetaWorldEnv into info so the
        # trainer can log episode aggregates: singularity, orientation fighting,
        # joint-velocity saturation, commanded vs achieved TCP motion).
        diag: dict[str, float] = {
            "cmd_norm": float(np.linalg.norm(e_pos)),
            "ori_capped": float(ori_capped),
        }
        try:
            diag["sigma_min"] = float(np.linalg.svd(J, compute_uv=False).min())
        except np.linalg.LinAlgError:
            diag["sigma_min"] = 0.0
        env._ctrl_diag = diag

        q_target = solve_dls(J, e, q, self._q_home, self._jnt_range, ik_cfg, diag=diag)

        # Guard: DLS result should be finite (bounded by joint clamp + λ).
        if not np.all(np.isfinite(q_target)):
            mujoco.mj_forward(m, d)
            return

        # --- gripper: action[3]=+1 → close (ctrl=0), -1 → open (ctrl=0.041) ---
        g_ctrl = float(_GRIPPER_MAX_OPEN * (1.0 - float(a[3])) / 2.0)

        # --- assemble full ctrl and simulate ---
        ctrl = d.ctrl.copy()
        ctrl[self._arm_act_ids] = q_target
        ctrl[self._grip_act_id] = g_ctrl
        env.do_simulation(ctrl, env.frame_skip)

    # ------------------------------------------------------------------
    # _external_reset_hand hook
    # ------------------------------------------------------------------

    def reset_hand(self, env: Any, steps: int) -> None:
        """Servo to home pose and set ``env.init_tcp``.

        ``env.init_tcp`` is REQUIRED by Meta-World reward functions
        (e.g., ``_gripper_caging_reward``, line 889 in sawyer_xyz_env.py).
        """
        assert self._arm_act_ids is not None
        assert self._grip_act_id is not None
        assert self._q_home is not None

        m: Any = env.model
        d: Any = env.data

        n_frames = max(steps, self._cfg.settle_steps)
        ctrl = d.ctrl.copy()
        ctrl[self._arm_act_ids] = self._q_home
        ctrl[self._grip_act_id] = _GRIPPER_MAX_OPEN  # open at reset
        env.do_simulation(ctrl, n_frames)
        mujoco.mj_forward(m, d)

        # tcp_center = mean of leftEndEffector / rightEndEffector sites
        env.init_tcp = env.tcp_center.copy()
