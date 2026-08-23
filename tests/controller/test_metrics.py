from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dreamer_arm.controller.metrics import ControllerMetrics


def test_controller_metrics_tracks_motion_from_reset_pose() -> None:
    data = SimpleNamespace(site_xpos=np.array([[0.0, 0.0, 0.0]]))
    metrics = ControllerMetrics(enabled=True, site_id=0)
    metrics.reset(data)
    data.site_xpos[0] = [0.01, 0.0, 0.0]
    metrics.accumulate({"cmd_norm": 0.02, "sigma_min": 0.3}, data)

    summary = metrics.summary()
    assert summary is not None
    assert summary["track_ratio_mean"] == pytest.approx(0.5)
    assert summary["frac_stuck"] == 0.0


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
