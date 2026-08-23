"""MuJoCo and Meta-World simulation backend."""

from dreamer_arm.envs.sim.factory import build_from_config, make_env, make_vector_env
from dreamer_arm.envs.sim.mujoco_logging import install_warning_handler

__all__ = ["build_from_config", "make_env", "make_vector_env"]

install_warning_handler()
