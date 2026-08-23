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
        self._lead_clamped = 0
        self._dq_max_sum = 0.0
        self._err_sum = 0.0
        self._track_sum = 0.0
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
        self._lead_clamped += int(diagnostics.get("lead_clamped", 0.0) > 0.0)
        self._dq_max_sum += float(diagnostics.get("dq_max", 0.0))
        self._err_sum += float(diagnostics.get("err_norm", 0.0))

        # Copy: site_xpos is a persistent MuJoCo buffer overwritten in place.
        tcp = np.asarray(data.site_xpos[self._site_id], dtype=np.float64).copy()
        cmd = float(diagnostics.get("cmd_norm", 0.0))
        if self._prev_tcp is not None and cmd > 1e-4:
            ratio = float(np.linalg.norm(tcp - self._prev_tcp)) / cmd
            self._track_sum += ratio
            self._track_n += 1
            self._stuck += int(ratio < 0.25)
        self._prev_tcp = tcp

    def summary(self) -> dict[str, float] | None:
        if not self._enabled or self._n == 0:
            return None
        return {
            "sigma_min_mean": self._sigma_sum / self._n,
            "sigma_min_min": self._sigma_min,
            "frac_ori_capped": self._ori_capped / self._n,
            "frac_dq_clamped": self._dq_clamped / self._n,
            "frac_lead_clamped": self._lead_clamped / self._n,
            "dq_max_mean": self._dq_max_sum / self._n,
            "err_norm_mean": self._err_sum / self._n,
            "track_ratio_mean": self._track_sum / self._track_n if self._track_n else 1.0,
            "frac_stuck": self._stuck / self._track_n if self._track_n else 0.0,
        }
