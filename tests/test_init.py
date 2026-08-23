from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("preset", [None, "", "  "])
def test_gl_backend_set_before_mujoco_is_imported(preset: str | None) -> None:
    env = dict(os.environ)
    if preset is None:
        env.pop("MUJOCO_GL", None)
    else:
        env["MUJOCO_GL"] = preset

    code = (
        "import builtins, os\n"
        "_real, seen = builtins.__import__, {}\n"
        "def spy(name, *a, **k):\n"
        "    if name.split('.')[0] == 'mujoco' and 'v' not in seen:\n"
        "        seen['v'] = os.environ.get('MUJOCO_GL', '')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = spy\n"
        "import dreamer_arm.core.buffer\n"
        "builtins.__import__ = _real\n"
        "print(repr(seen.get('v', '<never-imported>')))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    at_import = result.stdout.strip()
    expected = "'egl'" if sys.platform == "linux" else "'cgl'"
    if at_import == "'<never-imported>'":
        pytest.skip("torchrl no longer imports mujoco")
    assert at_import == expected


def test_explicit_gl_backend_is_respected() -> None:
    requested = "osmesa" if sys.platform == "linux" else "cgl"
    result = subprocess.run(
        [sys.executable, "-c", "import dreamer_arm; import os; print(os.environ['MUJOCO_GL'])"],
        env=dict(os.environ, MUJOCO_GL=requested),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == requested
