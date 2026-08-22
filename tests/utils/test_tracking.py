"""Tests for the W&B tracking wrapper (dreamer_arm.utils.tracking)."""

from __future__ import annotations

from typing import Any

import pytest
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


def test_scalars_stacks_each_device_group_separately(monkeypatch: Any) -> None:
    """A metrics dict spanning two devices must never reach one ungrouped torch.stack.

    torch.tensor(x) without device= defaults to CPU; mixed in with the rest of
    a Dreamer update's CUDA/MPS metrics (optim/step.py did exactly this until
    fixed alongside this test), a single ungrouped torch.stack raises

        RuntimeError: Tensor on device meta is not on the expected device cpu!

    (CUDA/MPS in production; reproduced here with 'meta', a real device
    available without an accelerator).  That failure only fires on a host with
    a second device, so it reached a real training run invisibly.

    'meta' tensors carry no data, so scalars() still fails on the *separate*
    meta-only group -- that .tolist() has nothing to read is expected and
    orthogonal to what this test checks. What must hold is that every
    torch.stack call the implementation makes is single-device, and that the
    raised error is that expected data-less-tensor error, not the
    mixed-device one above -- i.e. grouping happened before stacking, not after.
    """
    logger = _make_logger()
    stack_call_devices: list[set[torch.device]] = []
    original_stack = torch.stack

    def spy_stack(tensors: Any, *a: Any, **k: Any) -> Any:
        stack_call_devices.append({t.device for t in tensors})
        return original_stack(tensors, *a, **k)

    monkeypatch.setattr(torch, "stack", spy_stack)

    with pytest.raises(NotImplementedError, match="meta tensor"):
        logger.scalars(
            {
                "cpu_a": torch.tensor(1.0),
                "cpu_b": torch.tensor(2.0),
                "meta_a": torch.tensor(3.0, device="meta"),
            }
        )

    assert all(len(devices) == 1 for devices in stack_call_devices), stack_call_devices
    assert logger._scalars["cpu_a"] == 1.0
    assert logger._scalars["cpu_b"] == 2.0


def test_encode_video_returns_none_when_ffmpeg_missing(monkeypatch: Any) -> None:
    """A missing ffmpeg binary must drop the video, not crash the whole run.

    imageio.mimwrite raises RuntimeError("No ffmpeg exe could be found. ...")
    when imageio-ffmpeg has no usable binary -- e.g. an out-of-sync pixi env on
    a remote host.  Losing video logging for one run is fine; losing the run
    over an incomplete environment is not.
    """
    import numpy as np

    from dreamer_arm.utils import tracking

    def raise_no_ffmpeg(*a: Any, **k: Any) -> None:
        raise RuntimeError(
            "No ffmpeg exe could be found. Install ffmpeg on your system, "
            "or set the IMAGEIO_FFMPEG_EXE environment variable."
        )

    monkeypatch.setattr(tracking.imageio, "mimwrite", raise_no_ffmpeg)

    logger = _make_logger()
    frames = np.zeros((2, 3, 4, 4), dtype=np.uint8)  # (T, C, H, W)

    assert logger._encode_video(frames) is None
    assert logger._warned_no_ffmpeg is True


def test_write_skips_video_payload_when_ffmpeg_missing(monkeypatch: Any) -> None:
    logger = _make_logger()
    monkeypatch.setattr(logger, "_encode_video", lambda arr: None)
    logger.video("clip", torch.zeros(2, 3, 4, 4, dtype=torch.uint8))

    # write() must not raise, and must simply omit the video from the payload.
    logger.write(step=0)
