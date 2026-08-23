"""Route MuJoCo engine warnings through the project logger."""

from __future__ import annotations

import logging
import re

import mujoco

log = logging.getLogger(__name__)

_WARNING_TIME_SUFFIX = re.compile(r"\s*Time\s*=\s*[-\d.]+\.?\s*$")
_warning_counts: dict[str, int] = {}


def _forward_mujoco_warning(message: str) -> None:
    """Log engine warnings at widening intervals and suppress MUJOCO_LOG.TXT."""
    key = _WARNING_TIME_SUFFIX.sub("", message).strip()
    count = _warning_counts.get(key, 0) + 1
    _warning_counts[key] = count
    if count == 1:
        log.warning("mujoco: %s", message)
    elif count in (10, 100, 1000) or count % 10000 == 0:
        log.warning("mujoco: %s (seen %d times)", key, count)


def install_warning_handler() -> None:
    """Install the process-wide MuJoCo warning callback."""
    mujoco.set_mju_user_warning(_forward_mujoco_warning)
