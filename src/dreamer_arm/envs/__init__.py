"""Environment factory and wrappers for dreamer-arm.

Owns the capture of MuJoCo engine warnings.  The other piece of process-level
setup, choosing the GL backend, lives in dreamer_arm/__init__.py -- it has to
run before torchrl drags mujoco in, which happens before this module loads.
"""

from __future__ import annotations

import logging
import re

# MUJOCO_GL is set in dreamer_arm/__init__.py, not here: by the time this module
# runs, torchrl (via dreamer_arm.core.buffer) has usually imported mujoco
# already, and the backend is chosen at that first import.  See the comment
# there.  On CPU-only machines export MUJOCO_GL=osmesa before launching.
import mujoco

log = logging.getLogger(__name__)

#: MuJoCo appends "Time = 1.8000." to instability warnings; strip it so the same
#: warning at different sim times collapses to one key.
_WARNING_TIME_SUFFIX = re.compile(r"\s*Time\s*=\s*[-\d.]+\.?\s*$")

#: Times each distinct warning has been seen, so a diverging sim cannot flood
#: the log with one message per physics step.
_warning_counts: dict[str, int] = {}


def _forward_mujoco_warning(message: str) -> None:
    """Send a MuJoCo engine warning to the project logger.

    Registering any ``mju_user_warning`` handler also stops MuJoCo writing
    ``MUJOCO_LOG.TXT`` into the working directory: the default handler, which
    both appends to that file and prints to stderr, is bypassed entirely.  That
    is the point of this function -- instability warnings ("Nan, Inf or huge
    value in QPOS ...") belong in the run log next to the step lines that
    explain them, not in a stray file at the repo root that nobody reads.

    An unstable simulation emits the same warning on every step, so repeats are
    counted and reported at widening intervals instead of logged each time.
    """
    key = _WARNING_TIME_SUFFIX.sub("", message).strip()
    count = _warning_counts.get(key, 0) + 1
    _warning_counts[key] = count
    if count == 1:
        log.warning("mujoco: %s", message)
    elif count in (10, 100, 1000) or count % 10000 == 0:
        log.warning("mujoco: %s (seen %d times)", key, count)


mujoco.set_mju_user_warning(_forward_mujoco_warning)

from dreamer_arm.envs.factory import build_from_config, make_env, make_vector_env  # noqa: E402

__all__ = ["build_from_config", "make_env", "make_vector_env"]
