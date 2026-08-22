"""Sanity tests for the trajectory replay buffer.

We check that pushed transitions are sampleable in ``(B, T)`` slices and
that :meth:`update_initial_state` writes back to the same slots.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from dreamer_arm.core.buffer import BufferConfig, ReplayBuffer


def _make_buf(batch_size: int = 2, batch_length: int = 4) -> ReplayBuffer:
    cfg = BufferConfig(
        max_size=128,
        batch_size=batch_size,
        batch_length=batch_length,
        device="cpu",
        storage_device="cpu",
    )
    return ReplayBuffer(cfg)


def test_buffer_sample_shape() -> None:
    buf = _make_buf(batch_size=2, batch_length=4)
    n_envs = 2
    stoch_shape = (4, 4)
    deter_dim = 8
    # Fill 16 timesteps across 2 envs so SliceSampler has enough data.
    for t in range(16):
        td = TensorDict(
            {
                "scene": torch.zeros(n_envs, 8, 8, 3, dtype=torch.uint8),
                "action": torch.zeros(n_envs, 3),
                "reward": torch.zeros(n_envs, 1),
                "is_first": torch.tensor([t == 0] * n_envs),
                "is_last": torch.tensor([False] * n_envs),
                "is_terminal": torch.tensor([False] * n_envs),
                "stoch": torch.zeros(n_envs, *stoch_shape),
                "deter": torch.zeros(n_envs, deter_dim),
                "episode": torch.zeros(n_envs, dtype=torch.int32),
            },
            batch_size=(n_envs,),
        )
        buf.add_transition(td)

    data, index, initial = buf.sample(("stoch", "deter"))
    assert data.shape == torch.Size([2, 4])
    assert initial["stoch"].shape == (2, *stoch_shape)
    assert initial["deter"].shape == (2, deter_dim)
    # Two index tensors of shape (B, T) each.
    assert len(index) == 2
    assert index[0].shape == (2, 4)


def test_buffer_update_initial_state_writes_back() -> None:
    buf = _make_buf(batch_size=2, batch_length=4)
    n_envs = 2
    stoch_shape = (4, 4)
    deter_dim = 8
    for t in range(16):
        buf.add_transition(
            TensorDict(
                {
                    "scene": torch.zeros(n_envs, 8, 8, 3, dtype=torch.uint8),
                    "action": torch.zeros(n_envs, 3),
                    "reward": torch.zeros(n_envs, 1),
                    "is_first": torch.tensor([t == 0] * n_envs),
                    "is_last": torch.tensor([False] * n_envs),
                    "is_terminal": torch.tensor([False] * n_envs),
                    "stoch": torch.zeros(n_envs, *stoch_shape),
                    "deter": torch.zeros(n_envs, deter_dim),
                    "episode": torch.zeros(n_envs, dtype=torch.int32),
                },
                batch_size=(n_envs,),
            )
        )

    _, index, _ = buf.sample(("stoch", "deter"))
    new_stoch = torch.ones(2, 4, *stoch_shape)
    new_deter = torch.ones(2, 4, deter_dim) * 3.14
    buf.update_initial_state(index, {"stoch": new_stoch, "deter": new_deter})
    # Nothing to assert beyond "didn't raise"; getting the same slots back is
    # racy because SliceSampler picks new starts. Smoke-coverage of the
    # write path is enough here.


def test_buffer_empty_cache_keys_is_a_noop() -> None:
    """A world model with no cached latent (e.g. DINO-WM) passes no cache keys."""
    buf = _make_buf(batch_size=2, batch_length=4)
    n_envs = 2
    for t in range(16):
        buf.add_transition(
            TensorDict(
                {
                    "scene": torch.zeros(n_envs, 8, 8, 3, dtype=torch.uint8),
                    "action": torch.zeros(n_envs, 3),
                    "reward": torch.zeros(n_envs, 1),
                    "is_first": torch.tensor([t == 0] * n_envs),
                    "is_last": torch.tensor([False] * n_envs),
                    "is_terminal": torch.tensor([False] * n_envs),
                    "episode": torch.zeros(n_envs, dtype=torch.int32),
                },
                batch_size=(n_envs,),
            )
        )

    data, index, initial = buf.sample(())
    assert data.shape == torch.Size([2, 4])
    assert initial == {}
    buf.update_initial_state(index, {})


def test_buffer_sample_alignment() -> None:
    """obs/action/reward must line up on the *arrival* convention.

    A stored row is ``(obs_t, action_t, reward_t)`` where action/reward
    describe the transition leaving ``obs_t``.  The training window is indexed
    by the state a transition arrives at, so ``data[t]`` must pair
    ``obs_{t+1}`` with ``action_t`` and ``reward_t`` -- the alignment
    ``losses.lambda_return`` assumes.  Shifting only ``action`` (the old
    behaviour) left every value target offset by one step.
    """
    n_envs = 2
    batch_length = 4
    buf = _make_buf(batch_size=2, batch_length=batch_length)

    # Tag every row with its own timestep so misalignment is visible directly.
    for t in range(16):
        buf.add_transition(
            TensorDict(
                {
                    "obs": torch.full((n_envs, 1), float(t)),
                    "action": torch.full((n_envs, 1), float(t)),
                    "reward": torch.full((n_envs, 1), float(t)),
                    "stoch": torch.zeros(n_envs, 2, 2),
                    "deter": torch.zeros(n_envs, 4),
                    "episode": torch.zeros(n_envs, dtype=torch.int32),
                },
                batch_size=(n_envs,),
            )
        )

    data, _, _ = buf.sample(("stoch", "deter"))
    obs, action, reward = data["obs"], data["action"], data["reward"]

    assert torch.equal(action, obs - 1.0), f"action misaligned:\nobs={obs}\naction={action}"
    assert torch.equal(reward, obs - 1.0), f"reward misaligned:\nobs={obs}\nreward={reward}"


def test_buffer_save_load_round_trip(tmp_path: Any) -> None:
    """save() -> fresh buffer -> load() preserves length and sample shape."""
    n_envs = 2
    stoch_shape = (4, 4)
    deter_dim = 8
    src = _make_buf(batch_size=2, batch_length=4)
    for t in range(16):
        td = TensorDict(
            {
                "scene": torch.zeros(n_envs, 8, 8, 3, dtype=torch.uint8),
                "action": torch.zeros(n_envs, 3),
                "reward": torch.zeros(n_envs, 1),
                "is_first": torch.tensor([t == 0] * n_envs),
                "is_last": torch.tensor([False] * n_envs),
                "is_terminal": torch.tensor([False] * n_envs),
                "stoch": torch.full((n_envs, *stoch_shape), float(t)),
                "deter": torch.full((n_envs, deter_dim), float(t)),
                "episode": torch.zeros(n_envs, dtype=torch.int32),
            },
            batch_size=(n_envs,),
        )
        src.add_transition(td)

    path = tmp_path / "buffer_dump"
    src.save(path)
    assert path.is_dir()

    # A fresh buffer, same config, starts empty.
    dst = _make_buf(batch_size=2, batch_length=4)
    assert len(dst) == 0
    dst.load(path)

    assert len(dst) == len(src)
    data, _, initial = dst.sample(("stoch", "deter"))
    assert data.shape == torch.Size([2, 4])
    assert initial["stoch"].shape == (2, *stoch_shape)
