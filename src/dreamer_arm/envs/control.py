"""Damped-least-squares end-effector (EE) controller — 6-DOF pose servoing.

Maps a 4-D arm-agnostic action ``[Δx, Δy, Δz, gripper]`` to
position-actuator ``ctrl`` values via differential inverse kinematics (IK).

This is the single seam that makes the manipulation framework arm-swappable:
the same policy action means the same end-effector motion regardless of which
arm is underneath.  Swap the arm descriptor → the controller adapts.

Algorithm
---------
Given action ``a ∈ [-1, 1]^4``:

1. Compute desired EE displacement ``Δp = a[:3] * ee_step_m``.
2. Compute orientation error ``e_ori`` (world-frame, 3-vector) between the
   current TCP quaternion and the fixed target quaternion (captured once from
   the arm's ``home`` keyframe at construction time).
3. Build the 6xn TCP site Jacobian (translational + rotational rows) via
   ``mujoco.mj_jacSite``, restricted to the arm's DOFs.
4. Damped-least-squares (DLS) solution on the stacked 6-D error:
   ``dq = J^T (J J^T + λ²I₆)^{-1} [Δp; e_ori]``
5. Update position-actuator ctrl:
   ``ctrl[arm] = clip(q_arm + dq, ctrl_low, ctrl_high)``
6. Map gripper ``a[3] ∈ [-1, 1]`` → ``[gripper_closed, gripper_open]`` ctrl.

The orientation is **not** commanded by the policy; it is regulated to the
target pose defined by the home keyframe (typically gripper-down).

Position actuators drive joints to their ctrl target within the physics step;
no integration timestep arithmetic is needed here.
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("mujoco is required for the EEController") from exc

from dreamer_arm.envs.arms import Arm


class EEController:
    """Stateless (per-step) DLS end-effector controller.

    Parameters
    ----------
    arm:
        :class:`~dreamer_arm.envs.arms.Arm` descriptor for the arm.
    model:
        Compiled :class:`mujoco.MjModel` (with task bodies already spliced in).
    """

    def __init__(self, arm: Arm, model: mujoco.MjModel) -> None:
        self._arm = arm

        # TCP site index.
        self._tcp_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, arm.tcp_site))
        if self._tcp_id < 0:
            raise RuntimeError(f"TCP site {arm.tcp_site!r} not found in model (arm={arm.name!r})")

        # Arm joint → DOF / qpos address mappings.
        self._arm_dof_adrs: list[int] = []
        self._arm_qpos_adrs: list[int] = []
        for jname in arm.arm_joint_names:
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname))
            if jid < 0:
                raise RuntimeError(f"Arm joint {jname!r} not found in model (arm={arm.name!r})")
            self._arm_dof_adrs.append(int(model.jnt_dofadr[jid]))
            self._arm_qpos_adrs.append(int(model.jnt_qposadr[jid]))

        # Arm actuator IDs.
        self._arm_act_ids: list[int] = []
        for aname in arm.arm_actuator_names:
            aid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname))
            if aid < 0:
                raise RuntimeError(f"Arm actuator {aname!r} not found in model (arm={arm.name!r})")
            self._arm_act_ids.append(aid)

        # Gripper actuator ID.
        self._gripper_act_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, arm.gripper_actuator)
        )
        if self._gripper_act_id < 0:
            raise RuntimeError(
                f"Gripper actuator {arm.gripper_actuator!r} not found in model (arm={arm.name!r})"
            )

        # Actuator ctrl limits (needed to clamp joint targets).
        self._ctrl_low = model.actuator_ctrlrange[:, 0].astype(np.float64)
        self._ctrl_high = model.actuator_ctrlrange[:, 1].astype(np.float64)

        # Pre-allocate Jacobian buffers (3 x nv: translational + rotational).
        self._jacp = np.zeros((3, model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, model.nv), dtype=np.float64)

        # Target TCP orientation: use arm override or capture from home keyframe.
        if arm.tcp_target_quat is not None:
            self._quat_target = np.array(arm.tcp_target_quat, dtype=np.float64)
        else:
            scratch = mujoco.MjData(model)
            key_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))
            if key_id < 0:
                raise RuntimeError(
                    "No 'home' keyframe found; cannot capture target TCP orientation."
                )
            mujoco.mj_resetDataKeyframe(model, scratch, key_id)
            mujoco.mj_forward(model, scratch)
            self._quat_target = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(self._quat_target, scratch.site_xmat[self._tcp_id])

        # Scale factor for the orientation error term (relative to translation).
        self._ori_gain = 1.0

        # Cache scalars.
        self._lam2 = float(arm.ik_damping) ** 2
        self._step_m = float(arm.ee_step_m)
        self._g_lo = float(arm.gripper_closed)
        self._g_hi = float(arm.gripper_open)

    # ------------------------------------------------------------------ API

    def apply(
        self,
        action: np.ndarray,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        """Apply one 4-D EE action to ``data.ctrl``.

        Parameters
        ----------
        action:
            ``(4,)`` float array in ``[-1, 1]``: ``[Δx, Δy, Δz, gripper]``.
        model, data:
            Current MuJoCo model / data (after ``mj_forward`` or ``mj_step``).
        """
        delta_pos = np.asarray(action[:3], dtype=np.float64) * self._step_m

        # ---- 6-DOF DLS IK for arm joints ----
        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self._tcp_id)

        jacp_arm = self._jacp[:, self._arm_dof_adrs]  # (3, n_arm)
        jacr_arm = self._jacr[:, self._arm_dof_adrs]  # (3, n_arm)
        J = np.vstack([jacp_arm, jacr_arm])  # (6, n_arm)

        # Orientation error: world-frame angular velocity from current to target.
        quat_cur = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat_cur, data.site_xmat[self._tcp_id])
        e_ori = np.zeros(3, dtype=np.float64)
        mujoco.mju_subQuat(e_ori, self._quat_target, quat_cur)

        e = np.concatenate([delta_pos, self._ori_gain * e_ori])  # (6,)

        JJT = J @ J.T + self._lam2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, e)  # (n_arm,)

        q_cur = np.asarray(data.qpos)[self._arm_qpos_adrs]
        q_new = np.clip(
            q_cur + dq,
            self._ctrl_low[self._arm_act_ids],
            self._ctrl_high[self._arm_act_ids],
        )
        for i, aid in enumerate(self._arm_act_ids):
            data.ctrl[aid] = q_new[i]

        # ---- Gripper ----
        g_cmd = float(action[3])  # in [-1, 1]
        g_ctrl = self._g_lo + (g_cmd + 1.0) * 0.5 * (self._g_hi - self._g_lo)
        data.ctrl[self._gripper_act_id] = np.clip(g_ctrl, self._g_lo, self._g_hi)

    def tcp_pos(self, data: mujoco.MjData) -> np.ndarray:
        """Return the current TCP position as a float32 (3,) array."""
        return np.array(data.site_xpos[self._tcp_id], dtype=np.float32)

    def gripper_opening(self, data: mujoco.MjData) -> float:
        """Normalised gripper opening in [0, 1] (0 = closed, 1 = fully open)."""
        g_ctrl = float(data.ctrl[self._gripper_act_id])
        span = self._g_hi - self._g_lo
        if span == 0.0:
            return 0.0
        return float(np.clip((g_ctrl - self._g_lo) / span, 0.0, 1.0))
