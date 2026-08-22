"""Dreamer implementation for YAM robotic arms in MuJoCo."""

import os
import sys

# MuJoCo picks its GL backend once, at `import mujoco`, and never revisits it
# (mujoco/rendering/classic/gl_context.py reads MUJOCO_GL at module scope).  So
# the variable has to be set before *anything* pulls mujoco in, and that happens
# far earlier than it looks: dreamer_arm.core.buffer imports torchrl.data, which
# transitively imports mujoco via torchrl.envs.custom.  Setting it in
# dreamer_arm.envs was too late -- core.buffer is imported first, so the backend
# was already fixed by then.
#
# The failure is Linux-only and silent, which is why it survived: an unset
# MUJOCO_GL resolves to cgl on macOS (correct) but to *glfw* on Linux, which
# needs an X display and dies at mjr_makeContext with "an OpenGL platform
# library has not been loaded into this process".
#
# This module is the top-level package, so Python runs it before any
# dreamer_arm submodule regardless of entrypoint -- training, inference or
# pytest.  Keep it free of heavy imports.
#
# An empty or blank value counts as unset: mujoco treats "" as valid and falls
# through to glfw, so `MUJOCO_GL=` in a container image would silently
# reintroduce exactly the bug above.  Any real value the user set is preserved
# (e.g. MUJOCO_GL=osmesa for CPU-only boxes).
if not os.environ.get("MUJOCO_GL", "").strip():
    os.environ["MUJOCO_GL"] = "egl" if sys.platform == "linux" else "cgl"

__version__ = "0.1.0"
