"""Environment factory for dreamer-arm.

Parses the env_name string and returns a ``SyncVectorEnv`` ready for the
online trainer.

Name format
-----------
``"metaworld:<task>"``   - single-task MT1 (all envs share the task type)
``"metaworld:MT10"``     - multi-task MT10 (10 tasks, one pinned per env)
``"metaworld:MT25"``     - multi-task MT25 (25 tasks)
``"metaworld:MT50"``     - multi-task MT50 (50 tasks)

Multi-task constraint: ``env_num % task_count == 0``.

Task pinning: env ``i`` is permanently assigned to task type
``env_names[i % task_count]``.  Each env picks a fixed per-type task
(deterministic w.r.t. seed) so the agent never sees a task switch mid-episode.

The ``arm`` kwarg determines which arm plugin to install; ``set_active_arm``
is called **once per process** before constructing any Meta-World task env.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from typing import Any

import metaworld

from dreamer_arm.envs.arms import make_arm
from dreamer_arm.envs.metaworld import MetaWorldEnv
from dreamer_arm.envs.wrappers import ActionRatePenalty, SyncVectorEnv, TimeLimit

# Multi-task benchmark tags
_MT_TAGS: set[str] = {"MT10", "MT25", "MT50"}

# Map tag → metaworld benchmark class
_BENCHMARK_CLS: dict[str, type] = {
    "MT10": metaworld.MT10,
    "MT25": metaworld.MT25,
    "MT50": metaworld.MT50,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_env(
    env_name: str,
    seed: int,
    size: tuple[int, int],
    time_limit: int,
    action_repeat: int = 1,
    *,
    arm: str = "yam",
    camera: str = "corner",
    wrist_camera: str | None = None,
    camera_jitter: float = 0.0,
    wrist_fisheye: float = 0.0,
    scene_randomize: bool = False,
    camera_pose_randomize: bool = False,
    action_rate_cost: float = 0.0,
    action_mag_cost: float = 0.0,
    success_threshold: float = 1.0,
    # Multi-task pinning / viewer — set by make_vector_env
    _task: Any = None,
    _task_idx: int | None = None,
    _num_tasks: int | None = None,
    _env_cls: Any = None,
    _arm_obj: Any = None,
    _viewer: bool = False,
) -> Any:
    """Construct a single wrapped Meta-World Gymnasium env."""
    protocol, task_tag = _parse_name(env_name)
    assert protocol == "metaworld"

    # Determine the metaworld env class and task to use
    if _env_cls is not None:
        env_cls = _env_cls
        task = _task
    else:
        # MT1 single-task path
        mt1 = metaworld.MT1(task_tag, seed=seed)
        env_cls = next(iter(mt1.train_classes.values()))
        task = mt1.train_tasks[0]

    inner_env = env_cls(render_mode=None)

    # Attach arm (installs hooks)
    arm_obj = _arm_obj if _arm_obj is not None else make_arm(arm)
    arm_obj.attach(inner_env)

    wrapped = MetaWorldEnv(
        env=inner_env,
        task=task,
        arm=arm,
        size=size,
        camera=camera,
        wrist_camera=wrist_camera,
        scene_randomize=scene_randomize,
        camera_pose_randomize=camera_pose_randomize,
        camera_jitter=camera_jitter,
        wrist_fisheye=wrist_fisheye,
        task_idx=_task_idx,
        num_tasks=_num_tasks,
        success_threshold=success_threshold,
        viewer=_viewer,
    )
    wrapped = TimeLimit(wrapped, time_limit=time_limit, action_repeat=action_repeat)
    if action_rate_cost > 0.0 or action_mag_cost > 0.0:
        wrapped = ActionRatePenalty(
            wrapped,
            action_rate_cost=action_rate_cost,
            action_mag_cost=action_mag_cost,
        )
    return wrapped


def make_vector_env(
    env_name: str,
    num_envs: int,
    seed: int,
    size: tuple[int, int],
    action_repeat: int,
    time_limit: int,
    viewer: bool = False,
    **kwargs: Any,
) -> SyncVectorEnv:
    """Build a ``SyncVectorEnv`` with ``num_envs`` envs.

    ``env_name`` format: ``"metaworld:<task>"`` or ``"metaworld:MT10/25/50"``.

    All ``kwargs`` are forwarded to ``make_env`` (camera, DR flags, costs, …).
    The ``arm`` kwarg picks which arm plugin to use.

    Multi-task: each env is pinned to one task type; emits a one-hot
    ``task_id`` observation; requires ``num_envs % task_count == 0``.
    """
    protocol, task_tag = _parse_name(env_name)
    assert protocol == "metaworld", f"Unknown env protocol: {protocol!r}"

    arm_name: str = kwargs.pop("arm", "yam")
    # set_active_arm is process-global; safe to call multiple times with same value
    metaworld.set_active_arm(arm_name)

    # Resolve per-env (env_cls, task, task_idx, num_tasks) assignments
    assignments = _resolve_task_assignments(
        task_tag=task_tag,
        num_envs=num_envs,
        seed=seed,
    )

    # Build env_fns
    env_fns: list[Callable[[], Any]] = []
    for i, (env_cls, task, task_idx, _num_tasks) in enumerate(assignments):
        env_seed = seed + i

        # Capture loop vars by value
        def _make(
            _i: int = i,
            _env_cls: Any = env_cls,
            _task: Any = task,
            _task_idx: int | None = task_idx,
            _num_tasks_: int | None = _num_tasks,
            _env_seed: int = env_seed,
        ) -> Any:
            # One shared arm object per env so hooks don't cross-contaminate.
            # Viewer is only attached to env 0 (one window regardless of num_envs).
            arm_obj = make_arm(arm_name)
            return make_env(
                env_name=env_name,
                seed=_env_seed,
                size=size,
                time_limit=time_limit,
                action_repeat=action_repeat,
                arm=arm_name,
                _task=_task,
                _task_idx=_task_idx,
                _num_tasks=_num_tasks_,
                _env_cls=_env_cls,
                _viewer=viewer and (_i == 0),
                _arm_obj=arm_obj,
                **kwargs,
            )

        env_fns.append(_make)

    return SyncVectorEnv(env_fns, action_repeat=action_repeat)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_name(env_name: str) -> tuple[str, str]:
    """Split ``"metaworld:door-open"`` → ``("metaworld", "door-open")``."""
    parts = env_name.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid env_name {env_name!r}. Expected 'metaworld:<task>' or 'metaworld:MT10|MT25|MT50'.")
    return parts[0], parts[1]


def _resolve_task_assignments(
    task_tag: str,
    num_envs: int,
    seed: int,
) -> list[tuple[Any, Any, int | None, int | None]]:
    """Return a list of (env_cls, task, task_idx, num_tasks) for each env.

    For MT1: task_idx=None, num_tasks=None (no task_id obs key).
    For MT*: task_idx ∈ [0, task_count), num_tasks = task_count.
    """
    if task_tag in _MT_TAGS:
        bench_cls = _BENCHMARK_CLS[task_tag]
        bench = bench_cls(seed=seed)  # type: ignore[call-arg]
        train_classes: dict[str, Any] = dict(bench.train_classes)
        train_tasks: list[Any] = bench.train_tasks

        # Group tasks by env_name (preserving insertion order of train_classes)
        task_names: list[str] = list(train_classes.keys())
        task_count = len(task_names)

        if num_envs % task_count != 0:
            raise ValueError(
                f"env_num={num_envs} must be divisible by task_count={task_count} for benchmark {task_tag}."
            )

        tasks_by_name: dict[str, list[Any]] = collections.defaultdict(list)
        for t in train_tasks:
            tasks_by_name[t.env_name].append(t)

        assignments = []
        for i in range(num_envs):
            task_idx = i % task_count
            t_name = task_names[task_idx]
            env_cls = train_classes[t_name]
            # Pick one task from this type (deterministic per env index)
            avail = tasks_by_name[t_name]
            task = avail[i // task_count % len(avail)]
            assignments.append((env_cls, task, task_idx, task_count))
        return assignments

    else:
        # Single-task (MT1) — auto-append -v3 if the user wrote "door-open"
        if not task_tag.endswith("-v3"):
            task_tag = task_tag + "-v3"
        mt1 = metaworld.MT1(task_tag, seed=seed)
        env_cls = next(iter(mt1.train_classes.values()))
        # Distribute train_tasks round-robin across envs (up to 50 per type)
        all_tasks = mt1.train_tasks
        assignments = []
        for i in range(num_envs):
            task = all_tasks[i % len(all_tasks)]
            assignments.append((env_cls, task, None, None))
        return assignments


# Optional env kwargs, keyed by the config field that supplies them.  Only
# fields actually present in the active env group are forwarded, so each env
# config declares exactly the knobs it supports.
_OPTIONAL_ENV_FIELDS: dict[str, Callable[[Any], Any]] = {
    "success_threshold": float,
    "camera": str,
    "wrist_camera": str,
    "camera_jitter": float,
    "wrist_fisheye": float,
    "scene_randomize": bool,
    "camera_pose_randomize": bool,
    "action_rate_cost": float,
    "action_mag_cost": float,
}


def build_from_config(cfg: Any, *, viewer: bool = False) -> SyncVectorEnv:
    """Build a ``SyncVectorEnv`` from a resolved Hydra config.

    Shared by training and standalone evaluation so both construct envs the
    same way.  ``cfg`` is the *root* config: ``cfg.envs`` supplies the env
    group and the optional ``cfg.envs.arm`` group selects the arm plugin.
    """
    envs_cfg = cfg.envs
    extra: dict[str, Any] = {
        field: convert(envs_cfg[field])
        for field, convert in _OPTIONAL_ENV_FIELDS.items()
        if field in envs_cfg and envs_cfg[field] is not None
    }
    arm_cfg = envs_cfg.get("arm") if hasattr(envs_cfg, "get") else None
    if arm_cfg is not None:
        extra["arm"] = str(arm_cfg.name)

    return make_vector_env(
        f"{envs_cfg.name}:{envs_cfg.task}",
        num_envs=int(envs_cfg.env_num),
        seed=int(envs_cfg.seed),
        size=tuple(envs_cfg.size),
        action_repeat=int(envs_cfg.action_repeat),
        time_limit=int(envs_cfg.time_limit),
        viewer=viewer,
        **extra,
    )
