"""Environment backends and shared policy I/O contracts."""

from dreamer_arm.envs.action import ACTION_SPEC, ActionSpec
from dreamer_arm.envs.observation import ObservationSpec

__all__ = ["ACTION_SPEC", "ActionSpec", "ObservationSpec"]
