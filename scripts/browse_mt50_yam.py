"""Browse MT50 tasks with the YAM arm in the MuJoCo passive viewer.

Usage:
    pixi run python scripts/browse_mt50_yam.py              # all 50 tasks
    pixi run python scripts/browse_mt50_yam.py push hammer  # specific tasks

Controls:
    Close the viewer window  →  advance to the next task
    Ctrl-C in terminal       →  quit immediately

The scripted reach runs automatically so you can see the arm approach the
object. The arm repeats the reach loop until you close the window.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# macOS requires the viewer to run under mjpython (handles UI thread setup).
# Use an env-var guard to avoid an infinite re-exec loop (sys.executable
# inside mjpython doesn't always contain "mjpython" in its name).
if sys.platform == "darwin" and not os.environ.get("_IN_MJPYTHON"):
    mjpython = Path(sys.executable).parent / "mjpython"
    if mjpython.exists():
        os.environ["_IN_MJPYTHON"] = "1"  # inherited by the new process
        os.execv(str(mjpython), [str(mjpython), *sys.argv])
    else:
        sys.exit("mjpython not found — install mujoco or run: mjpython scripts/browse_mt50_yam.py")

import mujoco
import mujoco.viewer
import numpy as np

from dreamer_arm.envs.factory import _mt_task_names

SETTLE_STEPS = 10
MAX_REACH_STEPS = 200
RISE_Z = 0.40
ABOVE_OFFSET_M = 0.15
WAYPOINT_RADIUS = 0.02
STEP_SLEEP_S = 0.02  # seconds between steps (slower = easier to watch)


def _run_task(task: str) -> None:
    from dreamer_arm.envs.metaworld import MetaWorld

    print(f"\n{'=' * 60}")
    print(f"  Task: {task}")
    print(f"{'=' * 60}")

    env = MetaWorld(name=task, arm="yam", seed=0)
    env.reset()

    model = env._env.model
    data = env._env.data

    # Keep ori_gain at the training default (0.1) so the arm behaves as it
    # would during actual training — not zeroed like the verify script does.

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat[:] = [0.0, 0.65, 0.1]

        while viewer.is_running():
            # --- settle ---
            settle = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            for _ in range(SETTLE_STEPS):
                if not viewer.is_running():
                    break
                env.step(settle)
                viewer.sync()
                time.sleep(STEP_SLEEP_S)

            obj_pos = np.array(env._env._get_pos_objects()[:3], dtype=np.float64)
            init_tcp = np.asarray(env._env.tcp_center, dtype=np.float64)
            above_z = float(max(obj_pos[2] + ABOVE_OFFSET_M, RISE_Z))

            waypoints = [
                np.array([init_tcp[0], init_tcp[1], RISE_Z], dtype=np.float64),
                np.array([obj_pos[0], obj_pos[1], above_z], dtype=np.float64),
                obj_pos.copy(),
            ]
            wp_idx = 0

            # --- scripted reach ---
            for _ in range(MAX_REACH_STEPS):
                if not viewer.is_running():
                    break

                tcp = np.asarray(env._env.tcp_center, dtype=np.float64)

                while wp_idx < len(waypoints) - 1:
                    if float(np.linalg.norm(waypoints[wp_idx] - tcp)) < WAYPOINT_RADIUS:
                        wp_idx += 1
                    else:
                        break

                target = waypoints[wp_idx]
                delta = target - tcp
                dist = np.linalg.norm(delta)
                direction = np.clip(delta / (dist + 1e-8) * 5, -1, 1)

                env.step(np.array([*direction, -1.0], dtype=np.float32))
                viewer.sync()
                time.sleep(STEP_SLEEP_S)

                if float(np.linalg.norm(obj_pos - tcp)) < 0.08:
                    # Pause so you can see the success before looping
                    for _ in range(30):
                        if not viewer.is_running():
                            break
                        viewer.sync()
                        time.sleep(STEP_SLEEP_S)
                    break

            if not viewer.is_running():
                break

            # Loop: reset and try again so you can keep watching
            env.reset()
            wp_idx = 0

    env.close()


def main() -> None:
    if len(sys.argv) > 1:
        tasks = sys.argv[1:]
    else:
        tasks = _mt_task_names("MT50")

    print(f"Browsing {len(tasks)} task(s).  Close viewer window to advance.")

    for task in tasks:
        _run_task(task)

    print("\nDone.")


if __name__ == "__main__":
    main()
