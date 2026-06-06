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

from dataclasses import dataclass

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage
from torchrl.data.replay_buffers import ReplayBuffer as _TorchRLBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


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

    def sample(self) -> tuple[TensorDict, list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Draw a ``(B, T+1)`` slice, peel off the context step.

        Returns ``(data, index, initial)`` where:

        * ``data`` is the ``(B, T)`` training window. ``data["action"]`` is
          shifted one step back so that ``action[t]`` is the action that led
          into ``data[t]`` (matching the RSSM step semantics).
        * ``index`` is the ``(time_idx, env_idx)`` pair of the training
          window — pass these straight back to
          :meth:`update_initial_state`.
        * ``initial`` is ``(stoch, deter)`` for the slice's context step,
          ready to seed the RSSM rollout.
        """
        sample_td, info = self._buffer.sample(return_info=True)
        # SliceSampler returns flat (B*(T+1), ...); fold the time axis back in.
        sample_td = sample_td.view(-1, self.batch_length + 1)
        sample_td = self._move_to_device(sample_td)

        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        # action[t] is the action *taken at* t, so for the training window we
        # want the action that produced data[t] -- i.e. action[t-1] in the raw
        # slice, which is the prefix sample_td["action"][:, :-1].
        data.set_("action", sample_td["action"][:, :-1])
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def update_initial_state(
        self, index: list[torch.Tensor], stoch: torch.Tensor, deter: torch.Tensor
    ) -> None:
        """Write back the new RSSM latent for every sampled timestep.

        ``index`` is the pair returned by :meth:`sample`; ``stoch``/``deter``
        are the posterior latents predicted for those timesteps. Future
        samples that land on these slots will start their rollouts from this
        cached posterior instead of the zero prior.
        """
        flat_index = [ind.reshape(-1) for ind in index]
        stoch_flat = stoch.reshape(-1, *stoch.shape[2:])
        deter_flat = deter.reshape(-1, *deter.shape[2:])
        n = flat_index[0].shape[0]
        # Storage layout is (time, env); SliceSampler hands us indices in
        # (env_idx, time_idx) order, so swap.
        self._buffer[flat_index[1], flat_index[0]] = TensorDict(
            {"stoch": stoch_flat, "deter": deter_flat}, batch_size=(n,)
        )

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
