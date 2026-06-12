"""End-effector (EE) controller — 6-DOF pose servoing via damped-least-squares IK.

Maps a 4-D arm-agnostic action ``[Δx, Δy, Δz, gripper]`` to
position-actuator ``ctrl`` values via differential inverse kinematics (IK).

This is the single seam that makes the manipulation framework arm-swappable:
the same policy action means the same end-effector motion regardless of which
arm is underneath.  Swap the arm descriptor → the controller adapts.

Algorithm
---------
Given action ``a ∈ [-1, 1]^4``:

1. Compute desired EE displacement ``Δp = a[:3] * ee_step_m``.
2. Compute orientation error ``e_ori`` (world-frame, 3-vector).  In *down-axis*
   mode (``arm.tcp_approach_axis`` set) this is ``a_cur x a_tgt`` — it drives the
   gripper approach axis toward ``ori_target_axis`` (e.g. straight down) while
   leaving roll about that axis free, so the wrist is never pinned at its limits.
   Otherwise it is the quaternion error to a fixed captured target orientation.
   The gained error is clamped to ``max(|Δp|, 0.3 * ee_step_m)`` so an
   *unachievable* orientation (e.g. wrist at its limit) can never outweigh the
   commanded translation in the DLS objective — without the clamp the solver
   sacrifices position to chase orientation, dragging the TCP metres off target.
3. Build the 6xn TCP site Jacobian (translational + rotational rows) via
   ``mujoco.mj_jacSite``, restricted to the arm's DOFs.
4. Damped-least-squares solve on the stacked 6-D error, with Nakamura adaptive
   damping that ramps in near singularities:
   ``dq = Jᵀ (J Jᵀ + λ²I)⁻¹ [Δp; e_ori]``
5. Nullspace joint-limit avoidance: add ``N (k · (q_center - q))`` so redundant
   DOF(s) drift toward each joint's range centre, keeping joints off their limits.
6. Scale ``dq`` uniformly so ``max|dq_i| ≤ max_joint_step`` (step clamp).
7. Self-collision gate: if the candidate joint target puts any arm link into
   self-penetration, back off by halving ``dq`` (alpha = 0.5, 0.25) before
   falling back to holding the current pose.  Finger/gripper bodies are
   excluded from the check so normal gripper closing is never blocked.
8. Map gripper ``a[3] ∈ [-1, 1]`` → ``[gripper_closed, gripper_open]`` ctrl.

The orientation is **not** commanded by the policy; it is regulated toward the
target axis/orientation defined by the arm descriptor (typically gripper-down).

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

# Fraction of each joint's range used as the limit-avoidance margin: the
# nullspace bias is zero in the safe interior and ramps in only within this
# margin of a limit (so it never resists normal reaching motion).
_LIMIT_MARGIN_FRAC: float = 0.15

# Signed-distance threshold: contact.dist < this → treat as penetration.
# MuJoCo reports dist < 0 when surfaces overlap.  The arm's collision geoms are
# coarse capsules, so a few mm of capsule overlap is visually and physically
# harmless (arm joints are kinematically anchored — no force blow-up).  A small
# negative tolerance lets the arm *slide along* grazing configurations instead
# of freezing at them: with a strict 0.0 the twisted-wrist fold reached by
# constant (+1,-1,+1) commands vetoed every candidate step — including pure
# retreat — and locked the arm permanently.
_SELF_COLLISION_DIST: float = -0.003

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
    """Stateless (per-step) damped-least-squares end-effector controller.

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

        # Arm joint → DOF / qpos address mappings + range centres (used as the
        # neutral posture for the nullspace joint-limit-avoidance bias).
        self._arm_dof_adrs: list[int] = []
        self._arm_qpos_adrs: list[int] = []
        jnt_lo: list[float] = []
        jnt_hi: list[float] = []
        for jname in arm.arm_joint_names:
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname))
            if jid < 0:
                raise RuntimeError(f"Arm joint {jname!r} not found in model (arm={arm.name!r})")
            self._arm_dof_adrs.append(int(model.jnt_dofadr[jid]))
            self._arm_qpos_adrs.append(int(model.jnt_qposadr[jid]))
            lo_j, hi_j = model.jnt_range[jid]
            jnt_lo.append(float(lo_j))
            jnt_hi.append(float(hi_j))
        # Joint-limit-avoidance band: the nullspace bias is zero in the safe
        # interior [lo+margin, hi-margin] and repulsive only near a limit, so it
        # never resists normal reaching motion — it just keeps joints (notably the
        # wrist) off their hard limits.  Margin = _LIMIT_MARGIN_FRAC of each range.
        self._jnt_lo = np.array(jnt_lo, dtype=np.float64)
        self._jnt_hi = np.array(jnt_hi, dtype=np.float64)
        self._jnt_margin = _LIMIT_MARGIN_FRAC * (self._jnt_hi - self._jnt_lo)

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

        # Scale factor for the orientation error term (relative to translation).
        self._ori_gain = 1.0

        # Down-axis orientation regulation: regulate the gripper approach axis to
        # ``ori_target_axis`` (world frame), leaving roll about it free.  When
        # ``tcp_approach_axis`` is None we fall back to full-orientation tracking
        # of a fixed target quaternion (captured below).
        self._approach_axis = arm.tcp_approach_axis
        self._ori_target_axis = np.array(arm.ori_target_axis, dtype=np.float64)
        n = np.linalg.norm(self._ori_target_axis)
        if n > 0.0:
            self._ori_target_axis /= n

        # Fixed target TCP orientation — only needed in full-orientation mode.
        # Use the arm override, else capture from the 'home' keyframe.  Skipped
        # entirely in down-axis mode (the target is ori_target_axis, no quat).
        self._quat_target = np.zeros(4, dtype=np.float64)
        if self._approach_axis is None:
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
                mujoco.mju_mat2Quat(self._quat_target, scratch.site_xmat[self._tcp_id])

        # Cache scalars.
        self._max_joint_step = float(arm.max_joint_step)
        # DLS damping: base λ² plus Nakamura adaptive ramp up to λ²_max as
        # sigma_min falls below sigma0.  sigma0 == 0 disables the adaptive term.
        self._lam2_base = float(arm.ik_damping) ** 2
        self._lam2_max = float(arm.ik_damping_max) ** 2
        self._sigma0 = float(arm.ik_damping_sigma0)
        # Nullspace bias gain pulling the wrist off its limits (down-axis mode
        # leaves a 1-DOF nullspace; see apply()).
        self._null_gain = 0.1
        self._step_m = float(arm.ee_step_m)
        self._g_lo = float(arm.gripper_closed)
        self._g_hi = float(arm.gripper_open)

        # World-frame TCP workspace box (optional).  Outward motion at a face is
        # zeroed so the policy cannot push the TCP below the table, behind the
        # base, or out past the reachable workspace into the retract-fold.
        self._ws_lo: np.ndarray | None = None
        self._ws_hi: np.ndarray | None = None
        if arm.workspace_box is not None:
            self._ws_lo = np.array(arm.workspace_box[0], dtype=np.float64)
            self._ws_hi = np.array(arm.workspace_box[1], dtype=np.float64)

        # World-frame TCP reach sphere (optional), applied after the box: keeps
        # commanded targets inside the arm's healthy reach so the IK never
        # chases unreachable targets into the extension singularity.
        self._reach_center: np.ndarray | None = None
        self._reach_radius: float = 0.0
        if arm.reach_sphere is not None:
            self._reach_center = np.array(arm.reach_sphere[0], dtype=np.float64)
            self._reach_radius = float(arm.reach_sphere[1])

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
        # Keys: sigma_min, manip, lam2_eff, dq_norm, dq_max, clip_active,
        # backoff_alpha, cmd_norm, ws_clamp_active, near_limit.
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
        # Defensive clip: the controller contract is action ∈ [-1, 1] but the
        # actor's Normal samples are unbounded — never execute beyond design
        # velocity even if a caller forgets to clamp.
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        delta_pos = action[:3] * self._step_m

        # ---- Workspace clamp (box, then reach sphere) ----
        # Clamp the commanded TCP *target* (current pos + delta), then back out
        # the allowed delta.  Outward motion at a face becomes zero while
        # inward motion is untouched, so the policy can ride the boundary but
        # never cross it (no below-table, behind-base, or out past reach).
        _ws_clamp_active = False
        if self._ws_lo is not None or self._reach_center is not None:
            tcp_now = np.asarray(data.site_xpos[self._tcp_id], dtype=np.float64)
            target = tcp_now + delta_pos
            if self._ws_lo is not None:
                target = np.clip(target, self._ws_lo, self._ws_hi)
            if self._reach_center is not None:
                v = target - self._reach_center
                r = float(np.linalg.norm(v))
                if r > self._reach_radius:
                    target = self._reach_center + v * (self._reach_radius / r)
            clamped = target - tcp_now
            # Tolerance, not equality: (tcp + delta) - tcp loses low bits in
            # fp64, so exact comparison flagged ~every step as clamped (the
            # frac_ws_clamp metric read a useless 1.0).  1 µm ≫ roundoff and
            # ≪ any real clamping.
            _ws_clamp_active = bool(np.any(np.abs(clamped - delta_pos) > 1e-6))
            delta_pos = clamped
        # Post-clamp commanded TCP step (metres): the motion actually asked of
        # the IK.  Together with the achieved TCP displacement (measured by the
        # env after stepping) this drives stuck detection — wall-riding clamps
        # cmd_norm to ~0 so it is never misreported as a lock-up.
        self.last_diag["ws_clamp_active"] = float(_ws_clamp_active)
        self.last_diag["cmd_norm"] = float(np.linalg.norm(delta_pos))

        # ---- 6-DOF damped-least-squares IK for arm joints ----
        self._jacp[:] = 0.0
        self._jacr[:] = 0.0
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self._tcp_id)

        jacp_arm = self._jacp[:, self._arm_dof_adrs]  # (3, n_arm)
        jacr_arm = self._jacr[:, self._arm_dof_adrs]  # (3, n_arm)
        J = np.vstack([jacp_arm, jacr_arm])  # (6, n_arm)

        # ---- Orientation error ----
        if self._approach_axis is not None:
            # Down-axis regulation: rotate the gripper approach axis toward the
            # world target axis (e.g. straight down).  The cross product is a
            # small-angle rotation vector that is identically zero for any roll
            # about the approach axis, so wrist roll is left free — this is what
            # keeps the wrist off its ±limits at near-singular poses.
            site_mat = np.asarray(data.site_xmat[self._tcp_id]).reshape(3, 3)
            a_cur = site_mat[:, self._approach_axis]
            e_ori = np.cross(a_cur, self._ori_target_axis)
        else:
            # Full-orientation tracking of the captured target quaternion.
            quat_cur = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat_cur, data.site_xmat[self._tcp_id])
            e_ori = np.zeros(3, dtype=np.float64)
            mujoco.mju_subQuat(e_ori, self._quat_target, quat_cur)

        # Clamp the gained orientation term relative to the commanded step so an
        # unachievable orientation can never dominate the stacked error: position
        # tracking always wins, orientation regulates within that budget.  The
        # floor (0.3 * ee_step) keeps regulation alive for near-zero commands.
        e_ori_term = self._ori_gain * e_ori
        ori_cap = max(float(np.linalg.norm(delta_pos)), 0.3 * self._step_m)
        ori_norm = float(np.linalg.norm(e_ori_term))
        if ori_norm > ori_cap:
            e_ori_term *= ori_cap / ori_norm

        e = np.concatenate([delta_pos, e_ori_term])  # (6,)

        # ---- Damped-least-squares IK with adaptive damping ----
        svs = np.linalg.svd(J, compute_uv=False)  # min(6, n_arm) values
        sigma_min = float(svs.min())
        self.last_diag["sigma_min"] = sigma_min
        self.last_diag["manip"] = float(np.prod(svs))

        # Nakamura adaptive damping: λ²_eff ramps from λ²_base toward
        # λ²_base + λ²_max as sigma_min falls below sigma0.  Disabled when
        # sigma0 == 0 (constant base damping only).
        if self._sigma0 > 0.0 and sigma_min < self._sigma0:
            t = 1.0 - (sigma_min / self._sigma0) ** 2
            lam2 = self._lam2_base + t * self._lam2_max
        else:
            lam2 = self._lam2_base
        self.last_diag["lam2_eff"] = lam2

        # Single factorisation shared by the DLS solve and the nullspace
        # projector: M = (J Jᵀ + λ²I)⁻¹.
        M = np.linalg.solve(J @ J.T + lam2 * np.eye(6), np.eye(6))  # (6, 6)
        dq = J.T @ (M @ e)  # (n_arm,)

        # ---- Nullspace joint-limit avoidance ----
        # Repel each joint from its hard limits, but only within the avoidance
        # margin — the gradient is zero in the safe interior so it never resists
        # normal reaching.  Projected through N = I - J⁺J it perturbs only the
        # redundant DOF(s), keeping the wrist off its ±limits without changing
        # the commanded task motion.
        q_full = np.asarray(data.qpos)[self._arm_qpos_adrs]
        lo_violation = np.maximum(self._jnt_lo + self._jnt_margin - q_full, 0.0)
        hi_violation = np.minimum(self._jnt_hi - self._jnt_margin - q_full, 0.0)
        grad = lo_violation + hi_violation  # inward push, zero in the interior
        # Any joint inside its limit margin → limit-pinning diagnostic.  Logged
        # separately from sigma_min so limit lock-ups are distinguishable from
        # kinematic singularities.
        self.last_diag["near_limit"] = float(np.any(grad != 0.0))
        if self._null_gain > 0.0 and np.any(grad):
            N = np.eye(dq.shape[0]) - J.T @ (M @ J)
            dq = dq + N @ (self._null_gain * grad)

        # dq clamp: scale uniformly so max|dq_i| ≤ max_joint_step.
        # Preserves the IK direction; turns blowups into bounded smooth steps.
        if self._max_joint_step > 0.0:
            dq_peak = float(np.abs(dq).max())
            if dq_peak > self._max_joint_step:
                dq *= self._max_joint_step / dq_peak

        self.last_diag["dq_norm"] = float(np.linalg.norm(dq))
        self.last_diag["dq_max"] = float(np.abs(dq).max())

        q_cur = q_full

        # ---- Self-collision gate with back-off ----
        # Try alpha*dq for decreasing alpha; the first acceptable candidate wins.
        # A candidate is acceptable if it is penetration-free OR does not worsen
        # the *current* clearance — without the second condition a pose that is
        # already (even marginally) penetrating vetoes every step including pure
        # retreat, locking the arm permanently.  alpha=0.0 holds the current
        # pose (clearance identical by construction) and always wins, so q_new
        # is guaranteed to be set.
        cur_clearance = self._arm_self_clearance(q_cur)
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
            cand_clearance = self._arm_self_clearance(q_cand)
            if cand_clearance >= _SELF_COLLISION_DIST or cand_clearance >= cur_clearance:
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
        """Return True if ``q_candidate`` penetrates beyond the tolerance."""
        return self._arm_self_clearance(q_candidate) < _SELF_COLLISION_DIST

    def _arm_self_clearance(self, q_candidate: np.ndarray) -> float:
        """Worst signed distance among arm-link self-contact pairs at ``q_candidate``.

        Writes the 6 arm joint positions into a scratch :class:`mujoco.MjData`,
        runs a forward pass to compute collision geometry, then returns the most
        negative signed distance over contact pairs whose geoms both belong to
        arm link bodies (``+inf`` if there are none).  Values below
        :data:`_SELF_COLLISION_DIST` mean penetration.

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
        worst = float("inf")
        for i in range(scratch.ncon):
            c = scratch.contact[i]
            if int(c.geom1) in geoms and int(c.geom2) in geoms:
                worst = min(worst, float(c.dist))
        return worst
