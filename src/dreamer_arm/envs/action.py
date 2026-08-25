"""Backend-neutral policy action contract for simulation and hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Shared normalized Cartesian-rate and absolute-gripper action.

    The agent emits one float32 four-vector. Elements 0:3 are Cartesian tool
    velocity (x, y, z) in the controller's base/world frame, scaled by the
    backend controller's configured maximum speed. Element 3 is absolute
    gripper position: -1 fully open and +1 fully closed. Physical speed,
    workspace, and servo limits remain controller configuration rather than
    properties of the Gymnasium space.
    """

    DIM: ClassVar[int] = 4
    CARTESIAN: ClassVar[slice] = slice(0, 3)
    GRIPPER: ClassVar[int] = 3
    MIN: ClassVar[float] = -1.0
    MAX: ClassVar[float] = 1.0
    GRIPPER_OPEN: ClassVar[float] = -1.0
    GRIPPER_CLOSED: ClassVar[float] = 1.0

    def make_space(self) -> spaces.Box:
        """Return a fresh Gymnasium space for this contract."""
        return spaces.Box(self.MIN, self.MAX, (self.DIM,), dtype=np.float32)

    def coerce(self, action: Any) -> np.ndarray:
        """Validate and clip one backend-bound action as float32.

        Shape and finite-value failures are rejected because silently
        forwarding them to physical hardware is unsafe. Finite values are
        clipped to the declared bounds.
        """
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (self.DIM,):
            raise ValueError(f"action must have shape ({self.DIM},), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("action must contain only finite values")
        return np.clip(value, self.MIN, self.MAX)


ACTION_SPEC = ActionSpec()

__all__ = ["ACTION_SPEC", "ActionSpec"]
