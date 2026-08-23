"""Backend-independent damped-least-squares operational-space IK solver.

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
    """Per-action increment (metres) to the accumulated TCP setpoint when
    action magnitude == 1.0.  Not a per-step displacement bound — see
    ``max_lead_m``, which is what actually limits ‖e_pos‖ (and therefore
    ``dq``) at each control step."""

    damping: float = 0.10
    """DLS regularisation λ on the length-scaled task Jacobian (see
    ``length_scale``), where singular values are dimensionless and centred
    around O(1) (the YAM home spectrum's smallest value is ≈0.16).  λ² is
    added to the diagonal of J_w J_wᵀ, damping only directions near true
    kinematic degeneracy."""

    nullspace_gain: float = 1.0
    """Gain on the posture bias (q_home - q) projected into the null-space."""

    ori_gain: float = 1.0
    """Scale on the orientation error component."""

    joint_margin: float = 0.05
    """Soft-limit margin (rad) kept inside each joint limit."""

    max_joint_step: float = 1.0
    """Per-step cap on ``max|dq_i|`` (rad), applied separately to the task
    component and the nullspace posture-bias component before they are
    summed (so a small posture correction never steals clamp headroom from
    the commanded task motion, and vice versa — though the combined dq can
    reach up to 2x this in the worst case).  ``0`` disables the clamp."""

    max_lead_m: float = 0.25
    """Leash (metres) on how far the accumulated TCP setpoint may lead the
    actual TCP, i.e. a hard cap on ‖e_pos‖.  This — not ``ee_step_m`` — is
    what bounds the per-step task error (and therefore ``dq``) once the
    setpoint is integrated across steps rather than reset every step."""

    length_scale: float = 0.25
    """Characteristic length (metres) used to make the stacked position
    (metre-scale) and orientation (radian-scale, O(1)) rows of the task
    Jacobian unit-consistent before the DLS solve: position rows are divided
    by this.  Without it, ``sigma_min``/``damping`` are dominated by units,
    not kinematic conditioning."""


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

    ``site_xmat`` is the row-major matrix from ``data.site_xmat[id]`` and
    ``q_target`` uses the [w,x,y,z] convention.
    """
    q_cur = _mat2quat(site_xmat.reshape(3, 3))
    # q_err = q_target ⊗ conj(q_cur)
    q_cur_inv = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
    q_err = _mulquat(q_target, q_cur_inv)
    # Shortest-path: flip if w < 0 so |angle| ≤ π
    if q_err[0] < 0.0:
        q_err = -q_err
    return 2.0 * q_err[1:]


def _clamp_step(dq: np.ndarray, cap: float) -> tuple[np.ndarray, bool]:
    """Scale ``dq`` down uniformly (direction-preserving) so ``max|dq_i| <= cap``.

    ``cap <= 0`` disables the clamp.  Returns ``(dq, clamped)``.
    """
    if cap <= 0.0:
        return dq, False
    peak = float(np.abs(dq).max())
    if peak > cap:
        return dq * (cap / peak), True
    return dq, False


def solve_dls(
    J: np.ndarray,
    e: np.ndarray,
    q: np.ndarray,
    q_home: np.ndarray,
    jnt_range: np.ndarray,
    cfg: IKConfig,
    diag: dict[str, float] | None = None,
    sigma_min: bool = False,
) -> np.ndarray:
    """DLS IK with nullspace posture bias and unit-consistent Jacobian scaling.

    ``sigma_min`` gates the 6x6 SVD independently of the cheaper diagnostics.
    Training currently enables it for ``episode/ctrl_sigma_min_*``.
    """
    n = q.shape[0]

    # Weight the position rows by 1/length_scale so the position (m/rad) and
    # orientation (rad/rad, O(1)) blocks of J are on the same scale.  Without
    # this, sigma_min/damping are dominated by which units happen to be
    # smaller, not by kinematic conditioning (position singular values sit
    # ~O(0.05) at the YAM home pose purely from the metre/radian mismatch,
    # independent of joint configuration).
    w = np.ones(6)
    w[:3] = 1.0 / cfg.length_scale
    Jw = J * w[:, None]  # (6, n_arm)
    ew = e * w  # (6,)

    lam2 = cfg.damping**2
    JJt = Jw @ Jw.T + lam2 * np.eye(6)

    # One np.linalg.solve call against JJt for both right-hand sides (ew and
    # the identity) instead of two: LAPACK factorises JJt once and applies it
    # to every column of the combined (6, 7) right-hand side, rather than
    # factorising the same matrix twice.
    sol = np.linalg.solve(JJt, np.column_stack([ew, np.eye(6)]))  # (6, 7)
    x, JJt_inv = sol[:, 0], sol[:, 1:]  # (6,), (6, 6)

    dq_task = Jw.T @ x  # (n_arm,)

    Jdls = Jw.T @ JJt_inv  # (n_arm, 6) damped pseudo-inverse
    N = np.eye(n) - Jdls @ Jw  # (n_arm, n_arm)

    dq_null = N @ (cfg.nullspace_gain * (q_home - q))

    # Per-step joint-velocity clamp, applied to the task and posture-bias
    # components *separately* before summing.  A single clamp on their sum
    # would let an oversized task solve swamp the (much smaller) posture
    # bias entirely, or let a posture correction eat into clamp headroom the
    # task command needed — clamping each independently keeps both alive.
    dq_task, task_clamped = _clamp_step(dq_task, cfg.max_joint_step)
    dq_null, _null_clamped = _clamp_step(dq_null, cfg.max_joint_step)
    dq = dq_task + dq_null

    q_target = np.clip(
        q + dq,
        jnt_range[:, 0] + cfg.joint_margin,
        jnt_range[:, 1] - cfg.joint_margin,
    )

    if diag is not None:
        diag["dq_max"] = float(np.abs(dq).max())
        diag["dq_clamped"] = float(task_clamped)
        if sigma_min:
            try:
                diag["sigma_min"] = float(np.linalg.svd(Jw, compute_uv=False).min())
            except np.linalg.LinAlgError:
                diag["sigma_min"] = 0.0
    return q_target
