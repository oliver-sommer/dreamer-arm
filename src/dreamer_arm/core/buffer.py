"""TorchRL-backed replay buffer with cached RSSM initial states.

The Dreamer trainer samples ``(B, T+1)``-shaped trajectory slices: the first
timestep supplies the latent ``(stoch, deter)`` used as the RSSM initial
state, and the remaining ``T`` timesteps are the actual training window.

After each world-model update we overwrite the cached latent in the slots we
sampled from, so that future samples starting in the same window can resume
RSSM rollouts cheaply — the original R2-Dreamer trick. The reference repo
folded this caching into ``sample``/``update`` silently; here it lives behind
named methods (:meth:`sample`, :meth:`update_initial_state`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage
from torchrl.data.replay_buffers import ReplayBuffer as _TorchRLBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler

from dreamer_arm.utils.logging import adopt_logger

# This import is what pulls torchrl in, so it is also where torchrl's
# self-installed stdout handler has to be taken over.
adopt_logger("torchrl")


@dataclass(frozen=True)
class BufferConfig:
    """Sizing + device placement for the trajectory replay buffer."""

    max_size: int
    batch_size: int
    batch_length: int
    device: str = "cpu"
    storage_device: str = "cpu"
    episode_key: str = "episode"


class ReplayBuffer:
    """Trajectory replay buffer over a 2-D ``(time, env)`` storage grid.

    Storage layout follows TorchRL's ``LazyTensorStorage(ndim=2)``: the first
    axis is time, the second axis is the parallel-env index. Each
    :meth:`add_transition` call appends one timestep across all envs.
    Sampling returns ``(B, T+1)`` slices that always start on the boundary
    flagged by ``episode_key``.
    """

    def __init__(self, config: BufferConfig) -> None:
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self._episode_key = config.episode_key
        self._buffer = _TorchRLBuffer(
            storage=LazyTensorStorage(
                max_size=int(config.max_size),
                device=self.storage_device,
                ndim=2,
            ),
            sampler=SliceSampler(
                num_slices=self.batch_size,
                end_key=None,
                traj_key=self._episode_key,
                truncated_key=None,
                strict_length=True,
            ),
            prefetch=0,
            # +1 for the context step that feeds the RSSM initial state.
            batch_size=self.batch_size * (self.batch_length + 1),
        )

    # ------------------------------------------------------------------ writes

    def add_transition(self, data: TensorDict) -> None:
        """Append one batched timestep across all envs.

        ``data`` is shaped ``(num_envs, ...)``. We unsqueeze a time axis so
        the storage's ``(time, env)`` grid sees a single new row.
        """
        self._buffer.extend(data.unsqueeze(1))

    # ----------------------------------------------------------------- samples

    def sample(
        self, cache_keys: Sequence[str] = ("stoch", "deter")
    ) -> tuple[TensorDict, list[torch.Tensor], dict[str, torch.Tensor]]:
        """Draw a ``(B, T+1)`` slice, peel off the context step.

        Returns ``(data, index, initial)`` where:

        * ``data`` is the ``(B, T)`` training window. ``data["action"]`` and
          ``data["reward"]`` are shifted one step back so that ``action[t]``
          is the action that led into ``data[t]`` and ``reward[t]`` is the
          reward received on *arriving* there (matching the RSSM step
          semantics and DreamerV3's λ-return convention).
        * ``index`` is the ``(time_idx, env_idx)`` pair of the training
          window — pass these straight back to
          :meth:`update_initial_state`.
        * ``initial`` holds ``cache_keys`` (e.g. ``stoch``/``deter``) for the
          slice's context step, ready to seed the world model's rollout.
          Empty when ``cache_keys`` is empty (a world model with no cached
          latent, e.g. one whose state is a pure function of the observation).
        """
        sample_td, info = self._buffer.sample(return_info=True)
        # SliceSampler returns flat (B*(T+1), ...); fold the time axis back in.
        sample_td = sample_td.view(-1, self.batch_length + 1)
        sample_td = self._move_to_device(sample_td)

        initial = {key: sample_td[key][:, 0] for key in cache_keys}
        # A stored row is (obs_t, action_t, reward_t): action/reward describe the
        # transition *leaving* obs_t, so the arrival-indexed window takes both
        # from the preceding row.  This is the alignment `losses.lambda_return`
        # assumes; shifting only `action` offsets every value target by a step.
        # clone(): the source overlaps the view being written.
        shifted = {key: sample_td[key][:, :-1].clone() for key in ("action", "reward")}
        data = sample_td[:, 1:]
        for key, value in shifted.items():
            data.set_(key, value)
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        # tensordict's stubs widen indexing results to TensorCollection | Tensor.
        return data, index, initial  # ty: ignore[invalid-return-type]

    def update_initial_state(self, index: list[torch.Tensor], state: dict[str, torch.Tensor]) -> None:
        """Write back the new world-model latent for every sampled timestep.

        ``index`` is the pair returned by :meth:`sample`; ``state`` holds the
        posterior values (keyed the same as ``cache_keys``) predicted for
        those timesteps. Future samples that land on these slots will start
        their rollouts from this cached posterior instead of the zero prior.
        A no-op when ``state`` is empty.
        """
        if not state:
            return
        flat_index = [ind.reshape(-1) for ind in index]
        flat_state = {key: value.reshape(-1, *value.shape[2:]) for key, value in state.items()}
        n = flat_index[0].shape[0]
        # Storage layout is (time, env); SliceSampler hands us indices in
        # (env_idx, time_idx) order, so swap.
        self._buffer[flat_index[1], flat_index[0]] = TensorDict(flat_state, batch_size=(n,))

    # ---------------------------------------------------------------- introspect

    def __len__(self) -> int:
        shape = self._buffer.storage.shape
        if shape is None:
            return 0
        return int(shape.numel())

    # ----------------------------------------------------------------- internals

    def _move_to_device(self, td: TensorDict) -> TensorDict:
        src = td.device
        if src is None:
            return td
        if src.type == "cpu" and self.device.type == "cuda":
            return td.pin_memory().to(self.device, non_blocking=True)
        if src != self.device:
            return td.to(self.device, non_blocking=True)
        return td
