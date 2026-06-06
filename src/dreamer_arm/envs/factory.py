"""``make_env`` factory routing names to concrete env implementations.

Naming convention:

* ``"yam"`` or ``"yam:<task>"`` → :class:`dreamer_arm.envs.yam.YAM`.
* ``"dmc:<domain>_<task>"``    → :class:`dreamer_arm.envs.dmc.DeepMindControl`.

All envs are wrapped with :class:`DreamerObsWrapper` so the obs dict gets
``is_first``/``is_last``/``is_terminal`` flags, and (optionally) with
:class:`TimeLimit` if ``time_limit`` is set.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym

from dreamer_arm.envs.wrappers import DreamerObsWrapper, SyncVectorEnv, TimeLimit


def make_env(
    name: str,
    *,
    seed: int = 0,
    time_limit: int | None = None,
    size: tuple[int, int] = (64, 64),
    action_repeat: int = 1,
    viewer: bool = False,
    **kwargs: Any,
) -> gym.Env:  # type: ignore[type-arg]
    """Construct a single Dreamer-shaped env from a name.

    ``action_repeat`` is forwarded to the underlying env (which sums reward
    across sub-steps); ``time_limit`` is applied *after* the env so it
    counts wrapper-level steps, not physics sub-steps.

    ``viewer`` opens a passive MuJoCo window (YAM only; macOS + mjpython).
    """
    if name == "yam" or name.startswith("yam:"):
        from dreamer_arm.envs.yam import YAM

        task = name.split(":", 1)[1] if ":" in name else "reach"
        env: gym.Env = YAM(task=task, size=size, action_repeat=action_repeat, seed=seed, viewer=viewer, **kwargs)  # type: ignore[type-arg,arg-type]
    elif name.startswith("dmc:"):
        from dreamer_arm.envs.dmc import DeepMindControl

        env = DeepMindControl(
            name=name.split(":", 1)[1],
            action_repeat=action_repeat,
            size=size,
            seed=seed,
            **kwargs,
        )
    else:
        raise ValueError(f"unknown env name: {name!r}")

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

    Async vectorisation is intentionally out of scope (see plan).

    ``viewer=True`` opens a passive MuJoCo window on env 0 only.
    """
    def _factory(s: int, env_idx: int) -> Callable[[], gym.Env]:  # type: ignore[type-arg]
        def _make() -> gym.Env:  # type: ignore[type-arg]
            return make_env(name, seed=s, viewer=(viewer and env_idx == 0), **kwargs)

        return _make

    fns: list[Callable[[], gym.Env]] = [_factory(seed + i, i) for i in range(num_envs)]  # type: ignore[type-arg]
    return SyncVectorEnv(fns)
