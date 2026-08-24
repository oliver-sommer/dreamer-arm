from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dreamer_arm.envs.control.metrics import ControllerMetrics


def test_controller_metrics_tracks_motion_from_reset_pose() -> None:
    data = SimpleNamespace(site_xpos=np.array([[0.0, 0.0, 0.0]]))
    metrics = ControllerMetrics(enabled=True, site_id=0)
    metrics.reset(data)
    data.site_xpos[0] = [0.01, 0.0, 0.0]
    metrics.accumulate(
        {
            "cmd_norm": 0.02,
            "cmd_x": 0.02,
            "cmd_y": 0.0,
            "cmd_z": 0.0,
            "cmd_speed_m_s": 0.2,
            "ik_step_norm": 0.02,
            "err_x": 0.02,
            "err_y": 0.0,
            "err_z": 0.0,
            "sigma_min": 0.3,
            "ori_error_norm": 0.4,
            "ori_task_norm": 0.1,
        },
        data,
    )

    summary = metrics.summary()
    assert summary is not None
    assert summary["track_ratio_mean"] == pytest.approx(0.5)
    assert summary["frac_stuck"] == 0.0
    assert summary["ori_error_norm_mean"] == pytest.approx(0.4)
    assert summary["ori_task_norm_mean"] == pytest.approx(0.1)
    assert summary["cmd_x_mean"] == pytest.approx(0.02)
    assert summary["cmd_x_abs_mean"] == pytest.approx(0.02)
    assert summary["cmd_speed_m_s_mean"] == pytest.approx(0.2)
    assert summary["ik_step_norm_mean"] == pytest.approx(0.02)


def test_controller_metrics_does_not_count_orthogonal_drift_as_tracking() -> None:
    data = SimpleNamespace(site_xpos=np.array([[0.0, 0.0, 0.0]]))
    metrics = ControllerMetrics(enabled=True, site_id=0)
    metrics.reset(data)
    data.site_xpos[0] = [0.0, 0.02, 0.0]
    metrics.accumulate(
        {
            "cmd_norm": 0.02,
            "cmd_x": 0.02,
            "cmd_y": 0.0,
            "cmd_z": 0.0,
            "err_x": 0.02,
            "err_y": 0.0,
            "err_z": 0.0,
        },
        data,
    )

    summary = metrics.summary()
    assert summary is not None
    assert summary["track_ratio_mean"] == pytest.approx(0.0)
    assert summary["motion_ratio_mean"] == pytest.approx(1.0)
    assert summary["frac_stuck"] == 1.0


def test_controller_metrics_copies_mujoco_position_buffer() -> None:
    data = SimpleNamespace(site_xpos=np.array([[0.0, 0.0, 0.0]]))
    metrics = ControllerMetrics(enabled=True, site_id=0)
    metrics.reset(data)
    data.site_xpos[0] = [0.01, 0.0, 0.0]
    metrics.accumulate({"cmd_norm": 0.01}, data)
    data.site_xpos[0] = [0.02, 0.0, 0.0]
    metrics.accumulate({"cmd_norm": 0.01}, data)

    summary = metrics.summary()
    assert summary is not None
    assert summary["track_ratio_mean"] == pytest.approx(1.0)


def test_controller_metrics_disabled() -> None:
    data = SimpleNamespace(site_xpos=np.zeros((1, 3)))
    metrics = ControllerMetrics(enabled=False, site_id=-1)
    metrics.reset(data)
    metrics.accumulate({"sigma_min": 0.2}, data)
    assert metrics.summary() is None
