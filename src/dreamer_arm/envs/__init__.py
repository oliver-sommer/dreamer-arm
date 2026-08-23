"""Environment factory and wrappers for dreamer-arm."""

from dreamer_arm.envs.factory import build_from_config, make_env, make_vector_env
from dreamer_arm.envs.mujoco_logging import install_warning_handler

__all__ = ["build_from_config", "make_env", "make_vector_env"]

install_warning_handler()
