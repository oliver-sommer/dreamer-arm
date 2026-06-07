"""``make_env`` factory routing names to concrete env implementations.

Naming convention:

* ``"metaworld:<task>"``  → a single Meta-World MT1 task
  (:class:`dreamer_arm.envs.metaworld.MetaWorld`).
* ``"metaworld:MT10"`` / ``"MT25"`` / ``"MT50"``  → a multi-task generalist
  batch: one env pinned per task, each emitting a one-hot ``task_id`` for the
  policy to condition on.  Built by :func:`make_vector_env` →
  :func:`_make_metaworld_mt`.

All envs are wrapped with :class:`DreamerObsWrapper` so the obs dict gets
``is_first``/``is_last``/``is_terminal`` flags, and (optionally) with
:class:`TimeLimit` if ``time_limit`` is set.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from dreamer_arm.envs.wrappers import ActionRatePenalty, DreamerObsWrapper, SyncVectorEnv, TimeLimit

# Meta-World multi-task benchmarks → their ordered task-name lists.  The
# ``env_dict`` keys carry the ``-v3`` suffix; strip it to the bare task name
# that :class:`MetaWorld` expects.  Imported lazily in :func:`_mt_task_names`
# so importing this module does not pull in MuJoCo/Meta-World.
_MT_BENCHMARKS = ("MT10", "MT25", "MT50")


def _mt_task_names(benchmark: str) -> list[str]:
    """Ordered bare task names (no ``-v3``) for an MT benchmark."""
    from metaworld import env_dict

    dicts = {
        "MT10": env_dict.MT10_V3,
        "MT25": env_dict.MT25_V3,
        "MT50": env_dict.MT50_V3,
    }
    return [name.removesuffix("-v3") for name in dicts[benchmark]]


def make_env(
    name: str,
    *,
    seed: int = 0,
    time_limit: int | None = None,
    size: tuple[int, int] = (64, 64),
    action_repeat: int = 1,
    viewer: bool = False,
    action_rate_cost: float = 0.0,
    action_mag_cost: float = 0.0,
    **kwargs: Any,
) -> gym.Env:  # type: ignore[type-arg]
    """Construct a single Dreamer-shaped env from a name.

    ``action_repeat`` is forwarded to the underlying env (which sums reward
    across sub-steps); ``time_limit`` is applied *after* the env so it
    counts wrapper-level steps, not physics sub-steps.

    ``viewer`` opens a passive MuJoCo window (macOS + mjpython).
    ``arm`` selects which arm a Meta-World env uses (``"sawyer"`` or ``"yam"``).
    ``task_idx``/``num_tasks`` add a one-hot ``task_id`` obs key (multi-task).
    ``action_rate_cost`` / ``action_mag_cost`` add a smoothness/jerk penalty
    subtracted from the reward — recommended for YAM sim-to-real (~0.02).
    Both default to 0.0 (disabled) so benchmark runs are unaffected.
    """
    if name.startswith("metaworld:"):
        from dreamer_arm.envs.metaworld import MetaWorld

        arm = kwargs.pop("arm", "sawyer")  # forward arm to MetaWorld for YAM support
        env: gym.Env = MetaWorld(  # type: ignore[type-arg]
            name=name.split(":", 1)[1],
            arm=arm,
            action_repeat=action_repeat,
            size=size,
            seed=seed,
            viewer=viewer,
            task_idx=kwargs.pop("task_idx", None),
            num_tasks=kwargs.pop("num_tasks", None),
            **kwargs,  # forwards camera= from the env config
        )
    else:
        raise ValueError(f"unknown env name: {name!r}")

    if action_rate_cost or action_mag_cost:
        env = ActionRatePenalty(env, rate_cost=action_rate_cost, mag_cost=action_mag_cost)
    env = DreamerObsWrapper(env)
    if time_limit is not None:
        env = TimeLimit(env, max_steps=time_limit)
    return env


def make_vector_env(
    name: str,
    num_envs: int,
    *,
    seed: int = 0,
    viewer: bool = False,
    **kwargs: Any,
) -> SyncVectorEnv:
    """Build a synchronous batch of ``num_envs`` envs with offset seeds.

    Async vectorisation is intentionally out of scope.

    For a Meta-World multi-task benchmark (``"metaworld:MT10"`` etc.) this pins
    one task per env (env ``i`` → task ``i % num_tasks``) and tags each env with
    a one-hot ``task_id`` — see :func:`_make_metaworld_mt`.

    ``viewer=True`` opens a passive MuJoCo window on env 0 only.
    """
    if name.startswith("metaworld:") and name.split(":", 1)[1] in _MT_BENCHMARKS:
        return _make_metaworld_mt(
            name.split(":", 1)[1], num_envs, seed=seed, viewer=viewer, **kwargs
        )

    def _factory(s: int, env_idx: int) -> Callable[[], gym.Env]:  # type: ignore[type-arg]
        def _make() -> gym.Env:  # type: ignore[type-arg]
            return make_env(name, seed=s, viewer=(viewer and env_idx == 0), **kwargs)

        return _make

    fns: list[Callable[[], gym.Env]] = [_factory(seed + i, i) for i in range(num_envs)]  # type: ignore[type-arg]
    return SyncVectorEnv(fns)


def _make_metaworld_mt(
    benchmark: str,
    num_envs: int,
    *,
    seed: int = 0,
    viewer: bool = False,
    **kwargs: Any,
) -> SyncVectorEnv:
    """Build a multi-task Meta-World batch: one task pinned per env.

    Each env is a single-task :class:`MetaWorld` carrying a one-hot ``task_id``
    of length ``num_tasks``.  Env ``i`` runs task ``i % num_tasks``, so every
    gradient batch spans all tasks when ``num_envs`` is a multiple of the task
    count (``num_envs == num_tasks`` recommended).
    """
    task_names = _mt_task_names(benchmark)
    num_tasks = len(task_names)
    if num_envs % num_tasks != 0:
        raise ValueError(
            f"{benchmark} has {num_tasks} tasks; set envs.env_num to a multiple of "
            f"{num_tasks} (got {num_envs}) so each task is covered. "
            f"Recommended: envs.env_num={num_tasks}."
        )

    def _factory(env_idx: int) -> Callable[[], gym.Env]:  # type: ignore[type-arg]
        task_idx = env_idx % num_tasks
        name = f"metaworld:{task_names[task_idx]}"

        def _make() -> gym.Env:  # type: ignore[type-arg]
            return make_env(
                name,
                seed=seed + env_idx,
                viewer=(viewer and env_idx == 0),
                task_idx=task_idx,
                num_tasks=num_tasks,
                **kwargs,
            )

        return _make

    fns: list[Callable[[], gym.Env]] = [_factory(i) for i in range(num_envs)]  # type: ignore[type-arg]
    return SyncVectorEnv(fns)
