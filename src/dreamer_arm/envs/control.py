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
5. Compute the candidate joint target:
   ``q_cand = clip(q_arm + dq, ctrl_low, ctrl_high)``
6. Self-collision gate: if ``q_cand`` puts any arm link into self-penetration,
   back off by halving ``dq`` (alpha = 0.5, 0.25) before falling back to holding
   the current pose.  Finger/gripper bodies are excluded from the check so
   normal gripper closing is never blocked.
7. Map gripper ``a[3] ∈ [-1, 1]`` → ``[gripper_closed, gripper_open]`` ctrl.

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

# ---------------------------------------------------------------------------
# Self-collision gate
# ---------------------------------------------------------------------------

# Signed-distance threshold: contact.dist < this → treat as penetration.
# MuJoCo reports dist < 0 when surfaces overlap; 0.0 is the strict boundary.
_SELF_COLLISION_DIST: float = 0.0

# Back-off scale factors tried in order.  alpha=1.0 is the full IK step; alpha=0.0
# holds the current pose.  The first collision-free candidate wins.
_BACKOFF_ALPHAS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0)

# Bodies whose geoms participate in the self-collision check.  Finger/gripper
# bodies (leftclaw, rightclaw, lf_rot, lf_down, rf_rot, rf_down, leftpad,
# rightpad, hand) are intentionally omitted so that normal gripper closing and
# the fingers being near link_6 are never flagged as self-collision.
_ARM_LINK_BODY_NAMES: tuple[str, ...] = (
    "arm",
    "link_1",
    "link_2",
    "link_3",
    "link_4",
    "link_5",
    "link_6",
)


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
        self._lam2_base = float(arm.ik_damping) ** 2
        self._lam2_max = float(arm.ik_damping_max) ** 2
        self._sigma0 = float(arm.ik_damping_sigma0)
        self._max_joint_step = float(arm.max_joint_step)
        self._step_m = float(arm.ee_step_m)
        self._g_lo = float(arm.gripper_closed)
        self._g_hi = float(arm.gripper_open)

        # ---- Self-collision gate ----
        # Scratch MjData used for collision queries (never stepped; only used
        # with mj_forward to probe whether a candidate q causes arm link
        # self-penetration).
        self._model_ref = model
        self._scratch_col = mujoco.MjData(model)

        # Geom IDs whose body is one of the arm link bodies.  Built once from
        # the model; finger/hand bodies are excluded (see _ARM_LINK_BODY_NAMES).
        arm_link_geoms: set[int] = set()
        for bname in _ARM_LINK_BODY_NAMES:
            bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname))
            if bid >= 0:
                for gid in range(model.ngeom):
                    if int(model.geom_bodyid[gid]) == bid:
                        arm_link_geoms.add(gid)
        self._arm_link_geoms: frozenset[int] = frozenset(arm_link_geoms)

        # Per-step diagnostics populated by each call to apply().
        # Keys: sigma_min, manip, dq_norm, dq_max, clip_active, backoff_alpha.
        self.last_diag: dict[str, float] = {}

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

        # ---- SVD for diagnostics + adaptive damping (computed before solve) ----
        svs = np.linalg.svd(J, compute_uv=False)  # min(6, n_arm) values
        sigma_min = float(svs.min())
        self.last_diag["sigma_min"] = sigma_min
        self.last_diag["manip"] = float(np.prod(svs))

        # Nakamura adaptive damping: lam2_eff ramps from lam0^2 up to lam_max^2
        # as sigma_min falls below sigma_0.  Disabled when ik_damping_sigma0 == 0.
        if self._sigma0 > 0.0 and sigma_min < self._sigma0:
            t = 1.0 - (sigma_min / self._sigma0) ** 2
            lam2 = self._lam2_base + t * self._lam2_max
        else:
            lam2 = self._lam2_base
        self.last_diag["lam2_eff"] = lam2

        JJT = J @ J.T + lam2 * np.eye(6)
        dq = J.T @ np.linalg.solve(JJT, e)  # (n_arm,)

        # dq clamp: scale uniformly so max|dq_i| ≤ max_joint_step.
        # Preserves the IK direction; turns blowups into bounded smooth steps.
        if self._max_joint_step > 0.0:
            dq_peak = float(np.abs(dq).max())
            if dq_peak > self._max_joint_step:
                dq *= self._max_joint_step / dq_peak

        self.last_diag["dq_norm"] = float(np.linalg.norm(dq))
        self.last_diag["dq_max"] = float(np.abs(dq).max())

        q_cur = np.asarray(data.qpos)[self._arm_qpos_adrs]

        # ---- Self-collision gate with back-off ----
        # Try alpha*dq for decreasing alpha.  The first collision-free candidate wins.
        # alpha=0.0 holds the current (known collision-free) pose and always wins,
        # so q_new is guaranteed to be set.
        q_new = q_cur  # initialise to current pose as the final fallback
        _backoff_alpha = 0.0
        _clip_active = False
        for alpha in _BACKOFF_ALPHAS:
            q_unclamped = q_cur + alpha * dq
            q_cand = np.clip(
                q_unclamped,
                self._ctrl_low[self._arm_act_ids],
                self._ctrl_high[self._arm_act_ids],
            )
            if not self._self_collides(q_cand):
                q_new = q_cand
                _backoff_alpha = alpha
                _clip_active = bool(np.any(q_cand != q_unclamped))
                break
        self.last_diag["backoff_alpha"] = _backoff_alpha
        self.last_diag["clip_active"] = float(_clip_active)

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

    # ---------------------------------------------------------------- internals

    def _self_collides(self, q_candidate: np.ndarray) -> bool:
        """Return True if ``q_candidate`` causes any arm link self-penetration.

        Writes the 6 arm joint positions into a scratch :class:`mujoco.MjData`,
        runs a forward pass to compute collision geometry, then checks whether
        any contact pair has both geoms belonging to arm link bodies and a
        signed distance below :data:`_SELF_COLLISION_DIST`.

        Finger and hand geoms are excluded from the check (see
        :data:`_ARM_LINK_BODY_NAMES`), so gripper closing does not count as
        self-collision.

        Parameters
        ----------
        q_candidate:
            Array of 6 arm joint position targets (same order as
            ``arm.arm_joint_names``).
        """
        scratch = self._scratch_col
        for qpos_adr, q in zip(self._arm_qpos_adrs, q_candidate, strict=False):
            scratch.qpos[qpos_adr] = q
        mujoco.mj_forward(self._model_ref, scratch)
        geoms = self._arm_link_geoms
        for i in range(scratch.ncon):
            c = scratch.contact[i]
            if int(c.geom1) in geoms and int(c.geom2) in geoms and c.dist < _SELF_COLLISION_DIST:
                return True
        return False
