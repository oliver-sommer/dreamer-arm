"""Damped-least-squares (DLS) operational-space IK solver.

Pure-numpy, zero Meta-World coupling — importable in unit tests without MuJoCo.
All arrays are plain NumPy arrays; the caller supplies the Jacobian obtained
from ``mujoco.mj_jacSite``.

Design rationale (from the pitfall analysis):
- Constant-λ DLS bounds ``‖dq‖`` everywhere, even at singularities.  No
  workspace boxes, damping ramps, or stuck-detector required.
- Nullspace posture bias continuously pulls toward the interior home pose,
  repelling joint limits without velocity gating.
- Joint target clamping (not velocity gating) keeps the actuator live near
  limits, preventing the freeze symptom of the old design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IKConfig:
    """Tuning parameters for the DLS-IK controller."""

    ee_step_m: float = 0.05
    """Full TCP displacement (metres) when action magnitude == 1.0."""

    damping: float = 0.05
    """DLS regularisation λ.  λ² is added to the diagonal of J Jᵀ.
    With typical YAM singular values ≈ 0.05-1.8 m/rad, this damps only
    the near-degenerate directions (sigma ≲ λ) and caps ‖dq‖ ≤ ‖e‖ / (2λ)."""

    nullspace_gain: float = 1.0
    """Gain on the posture bias (q_home - q) projected into the null-space."""

    ori_gain: float = 1.0
    """Scale on the orientation error component."""

    joint_margin: float = 0.05
    """Soft-limit margin (rad) kept inside each joint limit."""

    max_joint_step: float = 0.15
    """Per-step cap on ``max|dq_i|`` (rad).  ``dq`` is scaled down uniformly when
    it exceeds this, preserving the IK direction while preventing a near-singular
    or orientation-dominated solve from slamming joints into their limits in a
    single control step.  ``0`` disables the clamp."""


# ---------------------------------------------------------------------------
# Quaternion helpers (pure numpy, [w, x, y, z] convention)
# ---------------------------------------------------------------------------


def _mat2quat(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix (shape (3,3) or row-major (9,)) → unit quaternion [w,x,y,z].

    Uses Shepperd's numerically stable method.
    """
    if mat.ndim == 1:
        mat = mat.reshape(3, 3)
    trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]))
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]))
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]))
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _mulquat(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Hamilton product qa ⊗ qb for unit quaternions [w,x,y,z]."""
    w1, x1, y1, z1 = qa
    w2, x2, y2, z2 = qb
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_log_error(site_xmat: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Orientation error as a rotation vector (3,).

    Computes ``2 * im(q_err)`` where ``q_err = q_target ⊗ q_cur⁻¹``,
    with sign normalised so that ``‖error‖ ≤ π``.

    Args:
        site_xmat: Row-major rotation matrix (9,) from ``data.site_xmat[id]``.
        q_target:  Unit quaternion [w,x,y,z] to servo toward.

    Returns:
        Rotation-vector error of shape (3,).
    """
    q_cur = _mat2quat(site_xmat.reshape(3, 3))
    # q_err = q_target ⊗ conj(q_cur)
    q_cur_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
    q_err = _mulquat(q_target, q_cur_inv)
    # Shortest-path: flip if w < 0 so |angle| ≤ π
    if q_err[0] < 0.0:
        q_err = -q_err
    return 2.0 * q_err[1:]


# ---------------------------------------------------------------------------
# DLS solver
# ---------------------------------------------------------------------------


def solve_dls(
    J: np.ndarray,
    e: np.ndarray,
    q: np.ndarray,
    q_home: np.ndarray,
    jnt_range: np.ndarray,
    cfg: IKConfig,
    diag: dict[str, float] | None = None,
) -> np.ndarray:
    """Damped-least-squares IK step with nullspace posture bias.

    Args:
        J:         (6, n_arm) Jacobian [Jp; Jr] sliced to arm DOF columns.
        e:         (6,) task error [e_pos (3); e_ori (3)].
        q:         (n_arm,) current arm joint angles (rad).
        q_home:    (n_arm,) home configuration to pull toward.
        jnt_range: (n_arm, 2) joint limits [[lo, hi], ...].
        cfg:       IK tuning config.
        diag:      Optional dict; when given, populated with ``dq_max`` and
                   ``dq_clamped`` for per-step controller diagnostics.

    Returns:
        (n_arm,) joint angle targets (already clamped to soft limits).
    """
    n = q.shape[0]
    lam2 = cfg.damping**2
    JJt = J @ J.T + lam2 * np.eye(6)

    # Particular solution: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
    x = np.linalg.solve(JJt, e)  # (6,)
    dq = J.T @ x  # (n_arm,)

    # Nullspace projector: N = I - Jᵀ (J Jᵀ + λ²I)⁻¹ J
    Jdls = J.T @ np.linalg.solve(JJt, np.eye(6))  # (n_arm, 6) damped pseudo-inverse
    N = np.eye(n) - Jdls @ J  # (n_arm, n_arm)

    # Posture bias: pull toward home pose
    dq = dq + N @ (cfg.nullspace_gain * (q_home - q))

    # Per-step joint-velocity clamp: scale dq uniformly so max|dq_i| ≤
    # max_joint_step.  Constant-λ DLS already bounds ‖dq‖, but the bound scales
    # with ‖e‖; an orientation-dominated or near-singular step can still command
    # a multi-radian joint jump that saturates against the limit clamp below and
    # parks the arm there.  Clamping dq direction-preservingly turns those into
    # bounded, smooth steps.
    dq_clamped = False
    if cfg.max_joint_step > 0.0:
        dq_peak = float(np.abs(dq).max())
        if dq_peak > cfg.max_joint_step:
            dq = dq * (cfg.max_joint_step / dq_peak)
            dq_clamped = True

    # Soft-limit clamping: clamp the commanded target, not the velocity
    q_target = np.clip(
        q + dq,
        jnt_range[:, 0] + cfg.joint_margin,
        jnt_range[:, 1] - cfg.joint_margin,
    )

    if diag is not None:
        diag["dq_max"] = float(np.abs(dq).max())
        diag["dq_clamped"] = float(dq_clamped)
    return q_target
