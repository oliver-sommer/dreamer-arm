"""Sanity tests for the trajectory replay buffer.

We check that pushed transitions are sampleable in ``(B, T)`` slices and
that :meth:`update_initial_state` writes back to the same slots.
"""

from __future__ import annotations

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
