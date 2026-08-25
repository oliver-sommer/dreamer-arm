"""Backend-independent controller state and per-episode tracking metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ServoState:
    """Resolved YAM indices and home targets used by the servo bench."""

    qpos_indices: np.ndarray
    actuator_ids: np.ndarray
    home_qpos: np.ndarray
    gripper_actuator_id: int


class ControllerMetrics:
    """Accumulate controller and achieved-motion metrics for one episode."""

    def __init__(self, enabled: bool, site_id: int) -> None:
        self._enabled = enabled
        self._site_id = site_id
        self.reset()

    def reset(self, data: Any | None = None) -> None:
        self._n = 0
        self._sigma_sum = 0.0
        self._sigma_min = float("inf")
        self._ori_capped = 0
        self._dq_clamped = 0
        self._joint_limit_clamped = 0
        self._joint_limit_clamped_by_joint = np.zeros(6, dtype=np.int64)
        self._dq_max_sum = 0.0
        self._ik_step_sum = 0.0
        self._target_lag_sum = 0.0
        self._lag_clamped = 0
        self._ws_clamped = 0
        self._cmd_speed_sum = 0.0
        self._cmd_axis_sum = np.zeros(3, dtype=np.float64)
        self._cmd_axis_abs_sum = np.zeros(3, dtype=np.float64)
        self._ori_error_sum = 0.0
        self._ori_task_sum = 0.0
        self._track_sum = 0.0
        self._motion_sum = 0.0
        self._track_n = 0
        self._stuck = 0
        self._prev_tcp = (
            np.asarray(data.site_xpos[self._site_id], dtype=np.float64).copy()
            if self._enabled and data is not None
            else None
        )

    def accumulate(self, diagnostics: Mapping[str, float] | None, data: Any) -> None:
        if not self._enabled or not diagnostics:
            return
        self._n += 1
        sigma = float(diagnostics.get("sigma_min", 0.0))
        self._sigma_sum += sigma
        self._sigma_min = min(self._sigma_min, sigma)
        self._ori_capped += int(diagnostics.get("ori_capped", 0.0) > 0.0)
        self._dq_clamped += int(diagnostics.get("dq_clamped", 0.0) > 0.0)
        self._joint_limit_clamped += int(diagnostics.get("joint_limit_clamped", 0.0) > 0.0)
        for i in range(6):
            self._joint_limit_clamped_by_joint[i] += int(diagnostics.get(f"joint_{i + 1}_limit_clamped", 0.0) > 0.0)
        self._dq_max_sum += float(diagnostics.get("dq_max", 0.0))
        self._ik_step_sum += float(diagnostics.get("ik_step_norm", 0.0))
        self._target_lag_sum += float(diagnostics.get("target_lag_norm", 0.0))
        self._lag_clamped += int(diagnostics.get("lag_clamped", 0.0) > 0.0)
        self._ws_clamped += int(diagnostics.get("ws_clamped", 0.0) > 0.0)
        self._cmd_speed_sum += float(diagnostics.get("cmd_speed_m_s", 0.0))
        for i, axis in enumerate("xyz"):
            cmd = float(diagnostics.get(f"cmd_{axis}", 0.0))
            self._cmd_axis_sum[i] += cmd
            self._cmd_axis_abs_sum[i] += abs(cmd)
        self._ori_error_sum += float(diagnostics.get("ori_error_norm", 0.0))
        self._ori_task_sum += float(diagnostics.get("ori_task_norm", 0.0))

        # Copy: site_xpos is a persistent MuJoCo buffer overwritten in place.
        tcp = np.asarray(data.site_xpos[self._site_id], dtype=np.float64).copy()
        cmd = float(diagnostics.get("cmd_norm", 0.0))
        if self._prev_tcp is not None and cmd > 1e-4:
            achieved = tcp - self._prev_tcp
            motion_ratio = float(np.linalg.norm(achieved)) / cmd
            cmd_keys = ("cmd_x", "cmd_y", "cmd_z")
            if all(key in diagnostics for key in cmd_keys):
                desired = np.array([diagnostics[key] for key in cmd_keys], dtype=np.float64)
                desired_norm = float(np.linalg.norm(desired))
                # Directional progress along the current velocity command.
                ratio = float(achieved @ desired) / (desired_norm * cmd) if desired_norm > 1e-12 else 0.0
            else:
                # Backward compatibility for third-party controllers that
                # only expose the scalar diagnostic.
                ratio = motion_ratio
            self._track_sum += ratio
            self._motion_sum += motion_ratio
            self._track_n += 1
            self._stuck += int(ratio < 0.25)
        self._prev_tcp = tcp

    def summary(self) -> dict[str, float] | None:
        if not self._enabled or self._n == 0:
            return None
        result = {
            "sigma_min_mean": self._sigma_sum / self._n,
            "sigma_min_min": self._sigma_min,
            "frac_ori_capped": self._ori_capped / self._n,
            "frac_dq_clamped": self._dq_clamped / self._n,
            "frac_joint_limit_clamped": self._joint_limit_clamped / self._n,
            "dq_max_mean": self._dq_max_sum / self._n,
            "ik_step_norm_mean": self._ik_step_sum / self._n,
            "target_lag_norm_mean": self._target_lag_sum / self._n,
            "frac_lag_clamped": self._lag_clamped / self._n,
            "frac_ws_clamped": self._ws_clamped / self._n,
            "cmd_speed_m_s_mean": self._cmd_speed_sum / self._n,
            "ori_error_norm_mean": self._ori_error_sum / self._n,
            "ori_task_norm_mean": self._ori_task_sum / self._n,
            "track_ratio_mean": self._track_sum / self._track_n if self._track_n else 1.0,
            "motion_ratio_mean": self._motion_sum / self._track_n if self._track_n else 1.0,
            "frac_stuck": self._stuck / self._track_n if self._track_n else 0.0,
        }
        for i, axis in enumerate("xyz"):
            result[f"cmd_{axis}_mean"] = self._cmd_axis_sum[i] / self._n
            result[f"cmd_{axis}_abs_mean"] = self._cmd_axis_abs_sum[i] / self._n
        for i, count in enumerate(self._joint_limit_clamped_by_joint, start=1):
            result[f"frac_joint_{i}_limit_clamped"] = float(count) / self._n
        return result
