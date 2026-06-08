"""Investigate YAM arm flip/jump/stuck behaviour in the full Meta-World environment.

The standalone singularity diagnostic showed the bare YAM IK is healthy, but the
MW model already has sigma_min=0.044 at the home pose (condition number 41.7 vs 13.1
standalone) — the wrist DOFs are more-singular in the spliced MW model.

This script runs the arm in the actual training env and looks for:
  - TCP position jumps  (> JUMP_THRESHOLD m in one step)
  - IK dq spikes        (dq_max > DQ_SPIKE_THRESHOLD rad)
  - sigma_min drops     (sigma_min < SIGMA_ALARM)
  - Wrist joint saturation (joint4/5 approaching ±π/2)
  - Arm config discontinuities (qpos jumps between consecutive steps)

Three stress scenarios per task:
  (A) Aggressive random walk — uniform(-1,1)^3 translation + open gripper
  (B) Wrist-stressing sweep  — large rotational / lateral oscillations designed
      to push joint4 and joint5 toward their ±π/2 limits
  (C) Limit-push             — repeatedly command the arm toward joint4 = +π/2
      by driving the EE in the direction that loads the wrist

Run with:
    pixi run python scripts/investigate_mw_physics.py
    pixi run python scripts/investigate_mw_physics.py --tasks reach push  # specific tasks
    pixi run python scripts/investigate_mw_physics.py --steps 500
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from dreamer_arm.envs.metaworld import MetaWorld

# ---- Thresholds -------------------------------------------------------
JUMP_THRESHOLD: float = 0.05  # m: TCP move > this in one env step = jump
DQ_SPIKE_THRESHOLD: float = 0.5  # rad: dq_max per step > this = spike
SIGMA_ALARM: float = 0.03  # sigma_min below this = near-singular alarm
WRIST_SAT_MARGIN: float = 0.05  # rad within joint limit = saturated

DEFAULT_STEPS: int = 300
DEFAULT_TASKS: list[str] = ["reach", "push", "pick-place", "window-open", "door-open"]


# ---- Parse args -------------------------------------------------------
_args = sys.argv[1:]
_steps_flag = DEFAULT_STEPS
_tasks_flag = DEFAULT_TASKS[:]
if "--steps" in _args:
    idx = _args.index("--steps")
    _steps_flag = int(_args[idx + 1])
if "--tasks" in _args:
    idx = _args.index("--tasks")
    # collect until next flag
    _tasks_flag = []
    i = idx + 1
    while i < len(_args) and not _args[i].startswith("--"):
        _tasks_flag.append(_args[i])
        i += 1


# ---- Data class for per-step event log --------------------------------
@dataclass
class StepEvent:
    step: int
    scenario: str
    tcp: np.ndarray
    tcp_jump: float
    dq_max: float
    sigma_min: float
    manip: float
    clip_active: bool
    backoff_alpha: float
    wrist_j4: float
    wrist_j5: float
    note: str = ""


@dataclass
class ScenarioStats:
    scenario: str
    n_steps: int = 0
    n_jumps: int = 0
    n_spikes: int = 0
    n_sigma_alarms: int = 0
    n_wrist_sat: int = 0
    n_clips: int = 0
    n_backoffs: int = 0
    sigma_min_values: list[float] = field(default_factory=list)
    dq_max_values: list[float] = field(default_factory=list)
    tcp_jump_values: list[float] = field(default_factory=list)
    worst_events: list[StepEvent] = field(default_factory=list)

    def add(self, ev: StepEvent) -> None:
        self.n_steps += 1
        self.sigma_min_values.append(ev.sigma_min)
        self.dq_max_values.append(ev.dq_max)
        self.tcp_jump_values.append(ev.tcp_jump)
        if ev.tcp_jump > JUMP_THRESHOLD:
            self.n_jumps += 1
        if ev.dq_max > DQ_SPIKE_THRESHOLD:
            self.n_spikes += 1
        if ev.sigma_min < SIGMA_ALARM:
            self.n_sigma_alarms += 1
        if abs(ev.wrist_j4) > (np.pi / 2 - WRIST_SAT_MARGIN) or abs(ev.wrist_j5) > (
            np.pi / 2 - WRIST_SAT_MARGIN
        ):
            self.n_wrist_sat += 1
        if ev.clip_active:
            self.n_clips += 1
        if ev.backoff_alpha < 1.0:
            self.n_backoffs += 1
        # Keep top-5 worst events by severity score
        severity = ev.tcp_jump * 2 + ev.dq_max + (1.0 / (ev.sigma_min + 1e-4))
        if len(self.worst_events) < 5 or severity > min(
            e.tcp_jump * 2 + e.dq_max + (1.0 / (e.sigma_min + 1e-4)) for e in self.worst_events
        ):
            self.worst_events.append(ev)
            self.worst_events.sort(
                key=lambda e: -(e.tcp_jump * 2 + e.dq_max + (1.0 / (e.sigma_min + 1e-4)))
            )
            self.worst_events = self.worst_events[:5]

    def summary(self) -> str:
        n = self.n_steps
        if n == 0:
            return "(no steps)"
        dq_arr = np.array(self.dq_max_values)
        sm_arr = np.array(self.sigma_min_values)
        jmp_arr = np.array(self.tcp_jump_values)
        return (
            f"smin: mean={sm_arr.mean():.4f} min={sm_arr.min():.4f}  "
            f"dq_max: max={dq_arr.max():.4f} mean={dq_arr.mean():.4f}  "
            f"tcp_jump max={jmp_arr.max() * 100:.1f}cm  "
            f"jumps={self.n_jumps}({self.n_jumps / n:.0%})  "
            f"spikes={self.n_spikes}({self.n_spikes / n:.0%})  "
            f"smin_alarms={self.n_sigma_alarms}({self.n_sigma_alarms / n:.0%})  "
            f"wrist_sat={self.n_wrist_sat}({self.n_wrist_sat / n:.0%})  "
            f"clips={self.n_clips}({self.n_clips / n:.0%})"
        )


# ---- Scenario action generators ---------------------------------------


def _random_actions(n: int, seed: int = 42) -> np.ndarray:
    """Return n x 4 uniform random actions in [-1,1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 4)).astype(np.float32)


def _wrist_stress_actions(n: int) -> np.ndarray:
    """Oscillate between ±y and ±x with large amplitude to stress wrist DOFs."""
    actions = np.zeros((n, 4), dtype=np.float32)
    period = 20
    for i in range(n):
        phase = (i % period) / period * 2 * np.pi
        # large ±y and ±x swings — drives joint4/5 hard
        actions[i, 0] = float(np.sin(phase))
        actions[i, 1] = float(np.cos(phase))
        actions[i, 2] = 0.1 * float(np.sin(2 * phase))
        actions[i, 3] = -1.0  # open gripper
    return actions


def _limit_push_actions(n: int) -> np.ndarray:
    """Repeatedly alternate between +y and -z to load joint4 toward +π/2."""
    actions = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        # Mostly +y (loads joint4); occasionally descend to change wrist config
        if i % 30 < 20:
            actions[i] = [0.0, 1.0, 0.0, -1.0]
        else:
            actions[i] = [0.0, -1.0, -0.5, -1.0]
    return actions


SCENARIOS: list[tuple[str, callable]] = [
    ("random_walk", lambda n: _random_actions(n)),
    ("wrist_stress", lambda n: _wrist_stress_actions(n)),
    ("limit_push", lambda n: _limit_push_actions(n)),
]


# ---- Per-step probe -----------------------------------------------


def _probe_step(
    env: MetaWorld,
    action: np.ndarray,
    prev_tcp: np.ndarray | None,
    step: int,
    scenario: str,
) -> StepEvent:
    """Take one step and record diagnostics."""
    _obs, _reward, _term, _trunc, _info = env.step(action)

    mw_env = env._env
    ctrl = mw_env._yam_controller
    d = ctrl.last_diag

    # TCP position
    tcp = ctrl.tcp_pos(mw_env.data).astype(np.float64)
    jump = float(np.linalg.norm(tcp - prev_tcp)) if prev_tcp is not None else 0.0

    # Wrist joint positions
    j4_idx = list(ctrl._arm.arm_joint_names).index("joint4")
    j5_idx = list(ctrl._arm.arm_joint_names).index("joint5")
    j4 = float(mw_env.data.qpos[ctrl._arm_qpos_adrs[j4_idx]])
    j5 = float(mw_env.data.qpos[ctrl._arm_qpos_adrs[j5_idx]])

    note = ""
    if jump > JUMP_THRESHOLD:
        note += f"JUMP({jump * 100:.1f}cm) "
    if d["dq_max"] > DQ_SPIKE_THRESHOLD:
        note += f"DQ_SPIKE({d['dq_max']:.3f}rad) "
    if d["sigma_min"] < SIGMA_ALARM:
        note += f"SINGULAR(smin={d['sigma_min']:.4f}) "

    return StepEvent(
        step=step,
        scenario=scenario,
        tcp=tcp.copy(),
        tcp_jump=jump,
        dq_max=d["dq_max"],
        sigma_min=d["sigma_min"],
        manip=d["manip"],
        clip_active=bool(d["clip_active"]),
        backoff_alpha=d["backoff_alpha"],
        wrist_j4=j4,
        wrist_j5=j5,
        note=note,
    )


# ---- Per-task investigation ----------------------------------------


def investigate_task(task: str, n_steps: int) -> None:
    print(f"\n{'═' * 72}")
    print(f"Task: {task}  ({n_steps} steps per scenario)")
    print(f"{'═' * 72}")

    for scenario_name, action_gen in SCENARIOS:
        try:
            env = MetaWorld(name=task, arm="yam", seed=0)
            _obs, _ = env.reset()
        except Exception as e:
            print(f"  [{scenario_name}]  SETUP ERROR: {e}")
            continue

        actions = action_gen(n_steps)
        stats = ScenarioStats(scenario=scenario_name)
        prev_tcp: np.ndarray | None = None

        mw_env = env._env
        ctrl = mw_env._yam_controller
        # Prime last_diag with a no-op step so it's populated from the start.
        ctrl.apply(np.zeros(4), mw_env.model, mw_env.data)

        notable: list[str] = []

        for i, a in enumerate(actions):
            ev = _probe_step(env, a, prev_tcp, i, scenario_name)
            stats.add(ev)
            prev_tcp = ev.tcp

            if ev.note:
                notable.append(f"    step {ev.note.strip()}")

            # Early-exit if physics explodes (TCP out of plausible workspace)
            if float(np.linalg.norm(ev.tcp)) > 2.0:
                notable.append(
                    f"    step {i}: PHYSICS EXPLODED (|tcp|={np.linalg.norm(ev.tcp):.2f}m)"
                )
                break

        env.close()

        print(f"\n  [{scenario_name}]")
        print(f"    {stats.summary()}")

        # Print worst events
        if stats.worst_events:
            print("    --- Top anomalies ---")
            for ev in stats.worst_events[:3]:
                if ev.tcp_jump > 0.01 or ev.dq_max > 0.2 or ev.sigma_min < 0.1:
                    print(
                        f"    step={ev.step:4d}  smin={ev.sigma_min:.4f}  "
                        f"dq_max={ev.dq_max:.4f}  jump={ev.tcp_jump * 100:.2f}cm  "
                        f"j4={ev.wrist_j4:.3f}  j5={ev.wrist_j5:.3f}  {ev.note}"
                    )

        if notable:
            print(f"    --- Notable steps ({len(notable)} total, first 10 shown) ---")
            for line in notable[:10]:
                print(line)
            if len(notable) > 10:
                print(f"    ... and {len(notable) - 10} more")


# ---- Entry point --------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("YAM in Meta-World: physics integration stress test")
    print(
        f"  Tasks: {_tasks_flag}\n"
        f"  Steps per scenario: {_steps_flag}\n"
        f"  Thresholds: jump>{JUMP_THRESHOLD * 100:.0f}cm  "
        f"dq_spike>{DQ_SPIKE_THRESHOLD}rad  smin_alarm<{SIGMA_ALARM}"
    )
    print("=" * 72)

    for task in _tasks_flag:
        try:
            investigate_task(task, _steps_flag)
        except Exception as e:
            print(f"\n[{task}] FATAL: {e}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
