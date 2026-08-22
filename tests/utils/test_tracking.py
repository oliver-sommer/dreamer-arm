"""Tests for the W&B tracking wrapper (dreamer_arm.utils.tracking)."""

from __future__ import annotations

from typing import Any

import torch

from dreamer_arm.utils.tracking import WandbLogger


def _make_logger() -> WandbLogger:
    return WandbLogger(project="dreamer-arm-test", mode="disabled")


def test_scalars_matches_individual_scalar_calls() -> None:
    """The batched path must record the same values as calling scalar() per key."""
    batched = _make_logger()
    batched.scalars({"a": torch.tensor(1.5), "b": torch.tensor(-2.0), "c": 3})

    individual = _make_logger()
    individual.scalar("a", torch.tensor(1.5))
    individual.scalar("b", torch.tensor(-2.0))
    individual.scalar("c", 3)

    assert batched._scalars == individual._scalars == {"a": 1.5, "b": -2.0, "c": 3.0}


def test_scalars_syncs_tensors_once_not_per_key(monkeypatch: Any) -> None:
    """scalars() must not call .item() per tensor -- that is the sync it exists to avoid.

    A Dreamer update reports 15-20 metrics; calling .item() once per key blocks
    on the device once per key.  scalars() should collapse that into a single
    torch.stack(...).tolist() regardless of how many tensors are passed.
    """
    logger = _make_logger()
    item_calls: list[int] = []
    original_item = torch.Tensor.item

    def counting_item(self: torch.Tensor) -> Any:
        item_calls.append(1)
        return original_item(self)

    monkeypatch.setattr(torch.Tensor, "item", counting_item)

    logger.scalars({f"k{i}": torch.tensor(float(i)) for i in range(20)})

    assert item_calls == [], ".item() was called; scalars() should use one stack().tolist() instead"
    assert logger._scalars == {f"k{i}": float(i) for i in range(20)}


def test_scalars_handles_mixed_tensor_and_plain_values() -> None:
    logger = _make_logger()
    logger.scalars({"loss": torch.tensor(0.5), "step": 100, "lr": 0.001})
    assert logger._scalars == {"loss": 0.5, "step": 100.0, "lr": 0.001}


def test_scalars_empty_is_a_noop() -> None:
    logger = _make_logger()
    logger.scalars({})
    assert logger._scalars == {}
