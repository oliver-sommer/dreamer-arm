"""Tests for `freeze_clone`: storage sharing across mutation, and grad blocking.

Every rollout/imagination path reads a `freeze_clone` of a trainable module
instead of the live one, on the assumption that in-place updates to the live
module (`p.copy_`, `load_state_dict`, an optimiser step) are immediately
visible through the clone without an explicit refresh -- only a *reallocating*
change (`.to(device)`) requires rebuilding the clone. These tests pin that
contract down directly, independent of any agent.
"""

from __future__ import annotations

import torch
from torch import nn

from dreamer_arm.core.frozen import freeze_clone


def test_freeze_clone_blocks_gradients() -> None:
    live = nn.Linear(3, 3)
    frozen = freeze_clone(live)

    for p in frozen.parameters():
        assert not p.requires_grad
    for p in live.parameters():
        assert p.requires_grad


def test_freeze_clone_reflects_in_place_updates_to_the_live_module() -> None:
    """An optimiser step (in-place) on `live` must be visible through `frozen`
    without calling `freeze_clone` again -- this is what lets `Dreamer` skip
    rebuilding frozen views after every `update()`, only after `.to()`.
    """
    live = nn.Linear(2, 2)
    frozen = freeze_clone(live)
    x = torch.randn(1, 2)

    with torch.no_grad():
        live.weight.copy_(torch.zeros_like(live.weight))
        live.bias.copy_(torch.zeros_like(live.bias))

    assert torch.equal(frozen.weight, live.weight)
    assert torch.allclose(frozen(x), torch.zeros(1, 2))


def test_freeze_clone_forward_matches_live_forward() -> None:
    torch.manual_seed(0)
    live = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
    frozen = freeze_clone(live)
    x = torch.randn(3, 4)
    assert torch.equal(live(x), frozen(x))
