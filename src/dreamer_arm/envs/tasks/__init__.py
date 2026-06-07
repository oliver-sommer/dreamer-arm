"""Task protocol and registry for the arm-swappable manipulation framework.

Each task is a class that implements :class:`Task` and is responsible for:

* **build** — splicing task-specific bodies (objects, goals, walls, etc.) into
  the compiled :class:`mujoco.MjSpec` *before* model compilation.
* **reset_ids** — caching body/mocap/site IDs from the compiled model once
  (called once per env construction after ``spec.compile()``).
* **reset** — re-randomising per-episode state (object spawn, goal position).
* **obs_keys** — the dict of additional observation keys this task contributes.
* **reward** — computing the (float reward, bool success) tuple.

Tasks reference the env only through its :class:`~dreamer_arm.envs.control.EEController`
(for TCP position) and the raw ``model``/``data`` objects — never arm internals —
so a task runs identically on any arm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import mujoco
import numpy as np

if TYPE_CHECKING:
    from dreamer_arm.envs.control import EEController


@runtime_checkable
class Task(Protocol):
    """Protocol that every manipulation task must implement."""

    name: str
    """Short identifier (e.g. ``"reach"``, ``"pick_place"``)."""

    obs_keys: tuple[str, ...]
    """Additional observation keys produced by :meth:`observe` (e.g.
    ``("target",)`` or ``("object", "goal")``).  Each value is a float32
    ``(3,)`` array."""

    def build(self, spec: mujoco.MjSpec) -> None:
        """Splice task-specific bodies into *spec* (called before compile)."""
        ...

    def reset_ids(self, model: mujoco.MjModel) -> None:
        """Cache body/site/mocap IDs from the compiled model (called once)."""
        ...

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
        """Randomise per-episode state (object spawn, goal, etc.)."""
        ...

    def observe(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Return task-specific obs entries (keys matching :attr:`obs_keys`)."""
        ...

    def reward(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller: EEController,
        success_threshold: float,
    ) -> tuple[float, bool]:
        """Return ``(reward, success)`` for the current state."""
        ...


def get_task(name: str) -> Task:
    """Return a fresh task instance for *name*."""
    if name == "reach":
        from dreamer_arm.envs.tasks.reach import ReachTask

        return ReachTask()
    if name == "pick_place":
        from dreamer_arm.envs.tasks.pick_place import PickPlaceTask

        return PickPlaceTask()
    raise ValueError(f"Unknown task: {name!r}.  Supported: 'reach', 'pick_place'.")
