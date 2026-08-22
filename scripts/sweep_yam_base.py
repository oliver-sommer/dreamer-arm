"""Sweep YAM arm-base placements against Meta-World task workspaces.

For each candidate base y-position, temporarily patches the ``arm`` body pos
in ``yam_xyz_base.xml``, then for every probed task servos the TCP to the
corners + centre of the task's goal space and object-spawn range and records
tracking error and the IK conditioning (sigma_min) on final approach.

A target counts as *covered* when the final error is < 2 cm and sigma_min
stays above 0.05 over the last 30 servo steps (i.e. reachable without
skirting a singularity).  The summary table reports covered/total per task
and per base — the basis for choosing the mounted position.

The original XML is always restored (try/finally), so this script is safe to
interrupt.

Run:  pixi run python scripts/sweep_yam_base.py [--bases 0.23 0.18 0.15] [--tasks reach ...]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

_XML = Path(__file__).resolve().parents[1] / "third_party/metaworld/metaworld/assets/objects/assets/yam_xyz_base.xml"

_DEFAULT_TASKS = (
    "reach",
    "push",
    "pick-place",
    "door-open",
    "drawer-open",
    "drawer-close",
    "button-press-topdown",
    "peg-insert-side",
    "window-open",
    "window-close",
)
_DEFAULT_BASES = (0.23, 0.20, 0.18, 0.15, 0.12)

_SERVO_STEPS = 150
_ERR_TOL = 0.02
_SIGMA_FLOOR = 0.05


def _box_targets(low: np.ndarray, high: np.ndarray) -> list[np.ndarray]:
    """Corners + centre of an axis-aligned box, deduplicated."""
    corners = [np.array(c) for c in itertools.product(*zip(low, high, strict=True))]
    corners.append((low + high) / 2.0)
    uniq: list[np.ndarray] = []
    for c in corners:
        if not any(np.allclose(c, u) for u in uniq):
            uniq.append(c)
    return uniq


def _task_targets(mw_env) -> list[np.ndarray]:
    """TCP probe targets for one task: goal space + object spawn range."""
    targets = _box_targets(
        np.asarray(mw_env.goal_space.low, dtype=np.float64),
        np.asarray(mw_env.goal_space.high, dtype=np.float64),
    )
    rrs = mw_env._random_reset_space
    targets += _box_targets(
        np.asarray(rrs.low[:3], dtype=np.float64),
        np.asarray(rrs.high[:3], dtype=np.float64),
    )
    # Clamp probe heights to a sane band: at least 1 cm above the table.
    return [np.array([t[0], t[1], max(t[2], 0.01)]) for t in targets]


def _probe_task(task: str) -> tuple[int, int, float]:
    """Return (covered, total, worst sigma_min among covered approaches)."""
    from dreamer_arm.envs.metaworld import MetaWorld

    env = MetaWorld(task, arm="yam", seed=0)
    ctrl = env._env._yam_controller
    d = env._env.data
    covered, total = 0, 0
    worst_sigma = np.inf
    for target in _task_targets(env._env):
        env.reset()
        sig_tail: list[float] = []
        for t in range(_SERVO_STEPS):
            tcp = ctrl.tcp_pos(d).astype(np.float64)
            a = np.clip((target - tcp) / 0.01, -1, 1)
            env.step(np.array([*a, -1.0], dtype=np.float32))
            if t >= _SERVO_STEPS - 30:
                sig_tail.append(ctrl.last_diag["sigma_min"])
        err = float(np.linalg.norm(ctrl.tcp_pos(d).astype(np.float64) - target))
        sig_min = float(np.min(sig_tail))
        total += 1
        if err < _ERR_TOL and sig_min > _SIGMA_FLOOR:
            covered += 1
            worst_sigma = min(worst_sigma, sig_min)
    env.close()
    return covered, total, worst_sigma


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=float, nargs="+", default=list(_DEFAULT_BASES))
    parser.add_argument("--tasks", type=str, nargs="+", default=list(_DEFAULT_TASKS))
    args = parser.parse_args()

    original = _XML.read_text()
    if 'pos="0 0.23 0.01"' not in original:
        raise SystemExit('expected arm body pos="0 0.23 0.01" in yam_xyz_base.xml')

    results: dict[float, dict[str, tuple[int, int, float]]] = {}
    try:
        for base_y in args.bases:
            _XML.write_text(original.replace('pos="0 0.23 0.01"', f'pos="0 {base_y} 0.01"'))
            results[base_y] = {}
            for task in args.tasks:
                try:
                    results[base_y][task] = _probe_task(task)
                except Exception as exc:  # noqa: BLE001 -- one task's failure must not abort the sweep
                    print(f"base_y={base_y} {task}: ERROR {type(exc).__name__}: {exc}")
                    results[base_y][task] = (0, 0, np.nan)
                cov, tot, _ = results[base_y][task]
                print(f"base_y={base_y:.2f} {task:22s} covered={cov}/{tot}")
    finally:
        _XML.write_text(original)

    print("\n=== summary (covered/total targets) ===")
    print(f"{'task':22s}" + "".join(f"  y={b:<7.2f}" for b in args.bases))
    for task in args.tasks:
        row = f"{task:22s}"
        for b in args.bases:
            cov, tot, _ = results[b].get(task, (0, 0, np.nan))
            row += f"  {cov:3d}/{tot:<5d}"
        print(row)
    print(f"{'TOTAL':22s}" + "".join(f"  {sum(c for c, _t, _s in results[b].values()):4d}     " for b in args.bases))


if __name__ == "__main__":
    main()
