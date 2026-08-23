"""MuJoCo engine warnings must reach the project logger, not a stray file.

Importing :mod:`dreamer_arm.envs.sim` registers an ``mju_user_warning`` handler,
which both routes warnings into ``logging`` and suppresses MuJoCo's default
``MUJOCO_LOG.TXT`` in the working directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np
import pytest

import dreamer_arm.envs.sim  # noqa: F401  -- import installs the warning handler
from dreamer_arm.envs.sim.mujoco_logging import _forward_mujoco_warning, _warning_counts

# Single sliding body: NaN in qpos makes mj_step emit the instability warning.
_XML = '<mujoco><worldbody><body><joint name="j" type="slide"/><geom size=".1"/></body></worldbody></mujoco>'


def _trigger_nan_warning() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    data.qpos[0] = np.nan
    mujoco.mj_step(model, data)


@pytest.fixture(autouse=True)
def _clear_counts():
    _warning_counts.clear()
    yield
    _warning_counts.clear()


def test_handler_is_installed() -> None:
    assert mujoco.get_mju_user_warning() is _forward_mujoco_warning


def test_engine_warning_is_logged(caplog: pytest.LogCaptureFixture, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING, logger="dreamer_arm.envs.sim.mujoco_logging"):
        _trigger_nan_warning()

    assert any("QPOS" in record.message for record in caplog.records), caplog.text
    # The whole point: no stray log file in the working directory.
    assert not (tmp_path / "MUJOCO_LOG.TXT").exists()


def test_repeated_warnings_are_counted_not_flooded(caplog: pytest.LogCaptureFixture) -> None:
    """A diverging sim warns every step; the log must not grow one line per step."""
    with caplog.at_level(logging.WARNING, logger="dreamer_arm.envs.sim.mujoco_logging"):
        for _ in range(20):
            _forward_mujoco_warning("Nan, Inf or huge value in QPOS at DOF 0. Time = 1.8000.")

    # First occurrence plus the 10th-repeat summary, not 20 lines.
    assert len(caplog.records) == 2
    assert "seen 10 times" in caplog.records[1].message


def test_differing_sim_times_collapse_to_one_key() -> None:
    """The trailing "Time = ..." varies every step and must not defeat dedup."""
    for t in ("1.8000", "1.8020", "1.8040"):
        _forward_mujoco_warning(f"Nan, Inf or huge value in QPOS at DOF 0. Time = {t}.")

    assert list(_warning_counts.values()) == [3]
