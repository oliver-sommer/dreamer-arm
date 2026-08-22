"""Structural contract the agent needs from a world model.

Three world models (categorical RSSM, DINO-WM, Dreamer 4) share the same
imagination-rollout and rollout-inference code in
:class:`~dreamer_arm.core.model.Dreamer` but differ completely in how they
represent state and compute their own representation loss. This module names
the small surface that is actually shared, so that surface can be dispatched
on uniformly while each model's ``loss`` stays free to do whatever is
structurally right for it.

State is an opaque ``dict[str, Tensor]``: the agent never inspects a state
dict's keys, it only round-trips whatever ``initial``/``img_step`` return.
Each implementation documents its own keys (RSSM: ``stoch``/``deter``).
"""

from __future__ import annotations

from typing import Protocol

import torch


class WorldModel(Protocol):
    """The subset of a world model's behaviour that is shared across models."""

    feat_size: int
    # State keys that should be cached in the replay buffer and written back
    # after each update (the R2-Dreamer latent-caching trick). Empty for
    # models whose state is a pure function of the observation, which need no
    # cache (e.g. DINO-WM).
    replay_cache_keys: tuple[str, ...]

    def initial(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Zero-initialised state for ``batch_size`` parallel rollouts."""
        ...

    def img_step(self, state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        """One-step prior transition, for imagination rollouts."""
        ...

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Flatten ``state`` into the feature vector the heads consume."""
        ...

    def encode_for_act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Embed one step of raw observation ``(B, ...)`` for rollout inference."""
        ...

    def observe_step(
        self,
        prev_state: dict[str, torch.Tensor],
        encoded: torch.Tensor,
        prev_action: torch.Tensor,
        is_first: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fold one freshly-encoded observation into the rollout state.

        Handles episode resets: implementations must zero whatever context
        they carry for a batch entry where ``is_first`` is set, so a fresh
        episode never sees the previous one's state.
        """
        ...

    def loss(
        self, data: dict[str, torch.Tensor], initial: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Compute this model's own representation-loss terms.

        ``data`` is the preprocessed training batch ``(B, T, ...)``;
        ``initial`` is the context state the sequence starts from (from
        :meth:`initial` or the replay buffer's cached ``replay_cache_keys``).

        Returns ``(state, losses, metrics)``:

        * ``state`` — the observed trajectory, one entry per state key, each
          ``(B, T, ...)``. Used both as imagination start states and (for
          ``replay_cache_keys``) written back to the buffer.
        * ``losses`` — this model's own loss terms only (e.g. ``dyn``/``rep``
          for RSSM, ``pred`` for DINO-WM). Reward/continue/actor/critic terms
          are computed by the agent from ``get_feat(state)`` and are not
          included here.
        * ``metrics`` — additional scalars to log as-is.
        """
        ...


__all__ = ["WorldModel"]
