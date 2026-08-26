"""Backend-independent controller state and per-episode tracking metrics."""

from __future__ import annotations

from collections import deque
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
    """Accumulate controller and achieved-motion metrics for one episode.

    Tracking is measured over a servo-timescale window rather than one 80 Hz
    physics-control interval.  The YAM shoulder servos need roughly 10--12
    intervals to settle; same-step action correlation mostly measures that
    expected latency and incorrectly reports a responsive arm as stuck.

    ``frac_stuck`` is deliberately stricter than directional under-tracking:
    it means the TCP path was less than 5% of the feasible reference path.
    Lag, cross-axis motion, and reversals remain visible through
    ``frac_undertracking`` and ``track_ratio_mean`` but are not physical stalls.
    """

    def __init__(self, enabled: bool, site_id: int, tracking_window_steps: int = 12) -> None:
        if tracking_window_steps < 1:
            raise ValueError("tracking_window_steps must be positive")
        self._enabled = enabled
        self._site_id = site_id
        self._tracking_window_steps = tracking_window_steps
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
        self._undertracking = 0
        self._path_ratio_sum = 0.0
        self._stall_n = 0
        self._stuck = 0
        initial_tcp = (
            np.asarray(data.site_xpos[self._site_id], dtype=np.float64).copy()
            if self._enabled and data is not None
            else None
        )
        self._tcp_window: deque[np.ndarray] = deque(maxlen=self._tracking_window_steps + 1)
        self._command_window: deque[np.ndarray] = deque(maxlen=self._tracking_window_steps)
        if initial_tcp is not None:
            self._tcp_window.append(initial_tcp)

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
        track_keys = ("track_cmd_x", "track_cmd_y", "track_cmd_z")
        cmd_keys = ("cmd_x", "cmd_y", "cmd_z")
        if all(key in diagnostics for key in track_keys):
            command = np.array([diagnostics[key] for key in track_keys], dtype=np.float64)
        elif all(key in diagnostics for key in cmd_keys):
            command = np.array([diagnostics[key] for key in cmd_keys], dtype=np.float64)
        else:
            # Scalar-only third-party controllers retain backward-compatible
            # motion-ratio semantics along an arbitrary fixed axis.
            command = np.array([float(diagnostics.get("cmd_norm", 0.0)), 0.0, 0.0])
        self._command_window.append(command)
        self._tcp_window.append(tcp)

        if (
            len(self._command_window) == self._tracking_window_steps
            and len(self._tcp_window) == self._tracking_window_steps + 1
        ):
            commands = np.asarray(self._command_window)
            tcp_positions = np.asarray(self._tcp_window)
            desired = np.sum(commands, axis=0)
            desired_norm = float(np.linalg.norm(desired))
            if desired_norm > 1e-4:
                achieved = self._tcp_window[-1] - self._tcp_window[0]
                motion_ratio = float(np.linalg.norm(achieved)) / desired_norm
                # Directional progress over the response window.  Net command
                # is used so rapid reversals that ask for no displacement do
                # not get mislabeled as a stuck controller.
                ratio = float(achieved @ desired) / (desired_norm * desired_norm)
                self._track_sum += ratio
                self._motion_sum += motion_ratio
                self._track_n += 1
                self._undertracking += int(ratio < 0.25)

            # A stall is a lack of physical motion, not merely poor alignment
            # with a rapidly changing reference.  Compare travelled paths so
            # a responsive reversal cannot look stationary just because the
            # TCP returns near its starting point.  Ignore sub-mm command
            # windows, where numerical/contact noise dominates the ratio.
            commanded_path = float(np.linalg.norm(commands, axis=1).sum())
            if commanded_path > 1e-3:
                achieved_path = float(np.linalg.norm(np.diff(tcp_positions, axis=0), axis=1).sum())
                path_ratio = achieved_path / commanded_path
                self._path_ratio_sum += path_ratio
                self._stall_n += 1
                self._stuck += int(path_ratio < 0.05)

    @property
    def tracking_samples(self) -> int:
        return self._track_n

    @property
    def stall_samples(self) -> int:
        return self._stall_n

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
            "frac_undertracking": self._undertracking / self._track_n if self._track_n else 0.0,
            "path_ratio_mean": self._path_ratio_sum / self._stall_n if self._stall_n else 1.0,
            "frac_stuck": self._stuck / self._stall_n if self._stall_n else 0.0,
            "track_sample_fraction": self._track_n / max(1, self._n - self._tracking_window_steps + 1),
            "stall_sample_fraction": self._stall_n / max(1, self._n - self._tracking_window_steps + 1),
        }
        for i, axis in enumerate("xyz"):
            result[f"cmd_{axis}_mean"] = self._cmd_axis_sum[i] / self._n
            result[f"cmd_{axis}_abs_mean"] = self._cmd_axis_abs_sum[i] / self._n
        for i, count in enumerate(self._joint_limit_clamped_by_joint, start=1):
            result[f"frac_joint_{i}_limit_clamped"] = float(count) / self._n
        return result
