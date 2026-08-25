"""Backend-neutral policy observation contract for simulation and hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Shared non-privileged camera, proprioception, and task observation.

    Proprioception has a fixed layout: six joint positions in radians,
    normalized gripper opening in [0, 1], and Cartesian position in metres of
    the tool point controlled by the backend, expressed in the controller's
    base/world frame. Images are uint8 RGB in HWC order. Simulation and
    hardware provide measurements; this class validates and packages them
    identically.
    """

    image_size: tuple[int, int]
    wrist_image: bool = False
    task_count: int | None = None

    SCENE: ClassVar[str] = "scene"
    WRIST_IMAGE: ClassVar[str] = "wrist_image"
    PROPRIO: ClassVar[str] = "proprio"
    TASK_ID: ClassVar[str] = "task_id"

    JOINT_DIM: ClassVar[int] = 6
    JOINTS: ClassVar[slice] = slice(0, 6)
    GRIPPER_OPEN: ClassVar[int] = 6
    TOOL_POSITION: ClassVar[slice] = slice(7, 10)
    TOOL_DIM: ClassVar[int] = 3
    PROPRIO_DIM: ClassVar[int] = 10
    IMAGE_CHANNELS: ClassVar[int] = 3

    def __post_init__(self) -> None:
        if len(self.image_size) != 2 or any(int(value) <= 0 for value in self.image_size):
            raise ValueError(f"image_size must contain two positive dimensions, got {self.image_size!r}")
        if self.task_count is not None and self.task_count <= 0:
            raise ValueError(f"task_count must be positive, got {self.task_count}")

    @property
    def image_shape(self) -> tuple[int, int, int]:
        height, width = (int(value) for value in self.image_size)
        return height, width, self.IMAGE_CHANNELS

    def make_space(self) -> spaces.Dict:
        """Return a fresh Gymnasium space for this configured contract."""
        values: dict[str, spaces.Space[Any]] = {
            self.SCENE: spaces.Box(0, 255, self.image_shape, dtype=np.uint8),
            self.PROPRIO: spaces.Box(-np.inf, np.inf, (self.PROPRIO_DIM,), dtype=np.float32),
        }
        if self.wrist_image:
            values[self.WRIST_IMAGE] = spaces.Box(0, 255, self.image_shape, dtype=np.uint8)
        if self.task_count is not None:
            values[self.TASK_ID] = spaces.Box(0.0, 1.0, (self.task_count,), dtype=np.float32)
        return spaces.Dict(values)

    def make(
        self,
        *,
        scene: Any,
        joint_positions: Any,
        gripper_open: Any,
        tool_position: Any,
        wrist_image: Any | None = None,
        task_index: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Validate backend measurements and create one policy observation."""
        observation = {
            self.SCENE: self._validate_image(scene, self.SCENE),
            self.PROPRIO: self.pack_proprio(joint_positions, gripper_open, tool_position),
        }
        if self.wrist_image:
            if wrist_image is None:
                raise ValueError("wrist_image is required by this observation contract")
            observation[self.WRIST_IMAGE] = self._validate_image(wrist_image, self.WRIST_IMAGE)
        elif wrist_image is not None:
            raise ValueError("wrist_image was provided but is not enabled")

        if self.task_count is not None:
            if task_index is None:
                raise ValueError("task_index is required by this observation contract")
            observation[self.TASK_ID] = self.make_task_id(task_index)
        elif task_index is not None:
            raise ValueError("task_index was provided but task identity is not enabled")
        return observation

    def pack_proprio(self, joint_positions: Any, gripper_open: Any, tool_position: Any) -> np.ndarray:
        """Pack backend measurements into the shared ten-value layout."""
        joints = np.asarray(joint_positions, dtype=np.float32)
        tool = np.asarray(tool_position, dtype=np.float32)
        if joints.shape != (self.JOINT_DIM,):
            raise ValueError(f"joint_positions must have shape ({self.JOINT_DIM},), got {joints.shape}")
        if tool.shape != (self.TOOL_DIM,):
            raise ValueError(f"tool_position must have shape ({self.TOOL_DIM},), got {tool.shape}")
        opening = float(gripper_open)
        if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(tool)) or not np.isfinite(opening):
            raise ValueError("proprioception must contain only finite values")

        proprio = np.empty(self.PROPRIO_DIM, dtype=np.float32)
        proprio[self.JOINTS] = joints
        proprio[self.GRIPPER_OPEN] = np.clip(opening, 0.0, 1.0)
        proprio[self.TOOL_POSITION] = tool
        return proprio

    def make_task_id(self, task_index: int) -> np.ndarray:
        """Return an exact one-hot task identity for this contract."""
        if self.task_count is None:
            raise ValueError("task identity is not enabled")
        if not 0 <= task_index < self.task_count:
            raise ValueError(f"task_index must be in [0, {self.task_count}), got {task_index}")
        task_id = np.zeros(self.task_count, dtype=np.float32)
        task_id[task_index] = 1.0
        return task_id

    def _validate_image(self, image: Any, key: str) -> np.ndarray:
        value = np.asarray(image)
        if value.shape != self.image_shape:
            raise ValueError(f"{key} must have shape {self.image_shape}, got {value.shape}")
        if value.dtype != np.uint8:
            raise ValueError(f"{key} must have dtype uint8, got {value.dtype}")
        return value


__all__ = ["ObservationSpec"]
