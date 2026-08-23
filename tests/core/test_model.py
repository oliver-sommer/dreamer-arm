"""End-to-end smoke tests for `Dreamer`: construct, act, fill a buffer, update.

Nothing here checks *learning* -- just that the composed agent (world model +
actor-critic + optimiser step) runs without shape errors and produces finite
losses, for both `wm` variants. `DinoEncoder` uses `pretrained=False` so this
needs no network access.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from dreamer_arm.core import Dreamer
from dreamer_arm.core.buffer import BufferConfig, ReplayBuffer
from dreamer_arm.utils.config import get_config_root, validate_config

_H = _W = 32
_N = 2


def _obs_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "scene": spaces.Box(0, 255, (_H, _W, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
        }
    )


def _act_space() -> spaces.Box:
    return spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)


def _compose(config_name: str, overrides: list[str]):  # type: ignore[no-untyped-def]
    with initialize_config_dir(config_dir=str(get_config_root()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    validate_config(cfg)
    return cfg


def _run_smoke(agent: Dreamer, batch_length: int) -> None:
    buffer = ReplayBuffer(
        BufferConfig(max_size=256, batch_size=_N, batch_length=batch_length, device="cpu", storage_device="cpu")
    )
    state = agent.get_initial_state(_N)
    cache_zeros = {k: torch.zeros_like(state[k]) for k in agent.replay_cache_keys}

    for t in range(batch_length * 3):
        obs = {
            "scene": torch.randint(0, 255, (_N, _H, _W, 3), dtype=torch.uint8),
            "proprio": torch.randn(_N, 7),
            "is_first": torch.tensor([t == 0] * _N),
        }
        action, next_state = agent.act(obs, state, eval_mode=False)
        assert torch.isfinite(action).all()

        td = TensorDict(
            {
                "scene": obs["scene"],
                "proprio": obs["proprio"],
                "action": action.detach(),
                "reward": torch.randn(_N, 1),
                "is_first": obs["is_first"],
                "is_last": torch.tensor([False] * _N),
                "is_terminal": torch.tensor([False] * _N),
                **cache_zeros,
                "episode": torch.zeros(_N, dtype=torch.int32),
            },
            batch_size=(_N,),
        )
        buffer.add_transition(td)
        state = next_state
        cache_zeros = {k: torch.zeros_like(state[k]) for k in agent.replay_cache_keys}

    for _ in range(2):
        mets = agent.update(buffer)
        for v in mets.values():
            assert torch.isfinite(torch.as_tensor(v)).all()


def test_rssm_agent_acts_and_updates() -> None:
    cfg = _compose(
        "training/dreamer",
        [
            "envs/sim=metaworld",
            "envs.sim.task=door-open",
            "envs.sim.size=[32,32]",
            "device=cpu",
            "core.model.compile=false",
        ],
    )
    agent = Dreamer(cfg.core.model, _obs_space(), _act_space())
    assert agent.replay_cache_keys == ("stoch", "deter")
    _run_smoke(agent, batch_length=8)


def test_dinowm_agent_acts_and_updates() -> None:
    cfg = _compose(
        "training/dreamer",
        [
            "core/model=dinowm",
            "envs/sim=metaworld",
            "envs.sim.task=door-open",
            "envs.sim.size=[32,32]",
            "core.model.dinowm.encoder.pretrained=false",
            "core.model.dinowm.encoder.image_size=32",
            "core.model.dinowm.context=3",
            "core.model.dinowm.predictor.depth=1",
            "core.model.dinowm.predictor.heads=2",
            "core.model.dinowm.predictor.dim_head=8",
            "core.model.dinowm.predictor.mlp_dim=16",
            "core.model.imag_starts=4",
            "core.model.imag_horizon=3",
            "device=cpu",
            "core.model.compile=false",
        ],
    )
    agent = Dreamer(cfg.core.model, _obs_space(), _act_space())
    assert agent.replay_cache_keys == ()
    _run_smoke(agent, batch_length=8)


def test_to_and_checkpoint_round_trip_preserve_frozen_views() -> None:
    """`.to()` and checkpoint load must rebuild frozen views from the moved/loaded weights."""
    cfg = _compose(
        "training/dreamer",
        [
            "envs/sim=metaworld",
            "envs.sim.task=door-open",
            "envs.sim.size=[32,32]",
            "device=cpu",
            "core.model.compile=false",
        ],
    )
    agent = Dreamer(cfg.core.model, _obs_space(), _act_space())
    agent.to("cpu")  # no-op device move, but exercises refresh_frozen()

    with torch.no_grad():
        for p in agent.wm_modules["rssm"].parameters():
            p.fill_(0.05)
    ckpt = agent.checkpoint_state()

    agent2 = Dreamer(cfg.core.model, _obs_space(), _act_space())
    agent2.load_checkpoint_state(ckpt)
    for p in agent2.frozen_wm.rssm.parameters():  # type: ignore[attr-defined]
        assert torch.allclose(p, torch.full_like(p, 0.05))


def test_compile_raises_recursion_limit() -> None:
    """Enabling compile must lift the recursion limit before inductor runs.

    Inductor's fusion pass walks the fused-node ancestor graph with a recursive
    DFS (``found_path`` inside ``Scheduler.will_fusion_create_cycle``).  One
    Dreamer update compiles as a single region deep enough to exceed CPython's
    default 1000 frames, and it fails *during compilation* with a bare
    RecursionError that names no dreamer_arm frame.

    Compile only runs on CUDA (``auto_compile``), so this cannot be reached by
    exercising the agent on a CPU/MPS test host; assert on the helper directly.
    """
    import sys

    from dreamer_arm.core.model import (
        _INDUCTOR_RECURSION_LIMIT,
        _raise_recursion_limit_for_inductor,
    )

    original = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(1000)
        _raise_recursion_limit_for_inductor()
        assert sys.getrecursionlimit() == _INDUCTOR_RECURSION_LIMIT

        # Must never lower a limit a caller deliberately set higher.
        sys.setrecursionlimit(_INDUCTOR_RECURSION_LIMIT + 5000)
        _raise_recursion_limit_for_inductor()
        assert sys.getrecursionlimit() == _INDUCTOR_RECURSION_LIMIT + 5000
    finally:
        sys.setrecursionlimit(original)
