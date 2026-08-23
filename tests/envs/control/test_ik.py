from __future__ import annotations

from typing import Any

import numpy as np

from dreamer_arm.envs.control.ik import IKConfig, _mat2quat, quat_log_error, solve_dls


def _make_cfg(**kwargs: Any) -> IKConfig:
    defaults: dict[str, Any] = {
        "ee_step_m": 0.05,
        "damping": 0.05,
        "nullspace_gain": 1.0,
        "ori_gain": 1.0,
        "joint_margin": 0.05,
    }
    defaults.update(kwargs)
    return IKConfig(**defaults)


def test_dls_bounded_near_singularity() -> None:
    rng = np.random.default_rng(0)
    q = np.zeros(6)
    jacobian = np.zeros((6, 6))
    jacobian[:, 0] = rng.normal(0, 1, 6)

    target = solve_dls(
        jacobian,
        np.ones(6) * 0.05,
        q,
        np.zeros(6),
        np.tile([-np.pi, np.pi], (6, 1)),
        _make_cfg(damping=0.05, nullspace_gain=0.0),
    )

    delta = target - q
    assert np.all(np.isfinite(delta))
    assert float(np.linalg.norm(delta)) < 10.0


def test_dls_posture_bias_toward_home() -> None:
    q = np.array([0.5, -0.3, 0.2, -0.1, 0.4, -0.2])
    home = np.zeros(6)
    target = solve_dls(
        np.random.default_rng(1).normal(0, 1, (6, 6)),
        np.zeros(6),
        q,
        home,
        np.tile([-np.pi, np.pi], (6, 1)),
        _make_cfg(nullspace_gain=1.0),
    )
    assert float(np.linalg.norm(home - target)) < float(np.linalg.norm(home - q))


def test_quat_log_error_identity() -> None:
    error = quat_log_error(np.eye(3).ravel(), np.array([1.0, 0.0, 0.0, 0.0]))
    assert error.shape == (3,)
    assert float(np.linalg.norm(error)) < 1e-6


def test_quat_log_error_nonzero() -> None:
    matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    _mat2quat(matrix)
    error = quat_log_error(matrix.ravel(), np.array([1.0, 0.0, 0.0, 0.0]))
    assert float(np.linalg.norm(error)) > 0.5


def test_solve_dls_joint_clamp() -> None:
    margin = 0.1
    limits = np.tile([-1.5, 1.5], (6, 1))
    target = solve_dls(
        np.eye(6),
        np.ones(6) * 0.5,
        np.array([1.4, -1.4, 1.4, -1.4, 1.4, -1.4]),
        np.zeros(6),
        limits,
        _make_cfg(joint_margin=margin),
    )
    assert np.all(target >= limits[:, 0] + margin - 1e-9)
    assert np.all(target <= limits[:, 1] - margin + 1e-9)
