from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import torch

from dreamer_arm.training.checkpoint import CheckpointManager


class _Agent:
    def checkpoint_state(self) -> dict[str, Any]:
        return {"weight": torch.tensor([1.0])}


def test_checkpoint_payload_and_archive(tmp_path: Any) -> None:
    buffer = MagicMock()
    manager = CheckpointManager(tmp_path, buffer, checkpoint_buffer=True)
    manager.save(_Agent(), 42, {"next_ep_id": 7}, archive=True)

    latest = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert set(latest) == {"agent", "step", "trainer"}
    assert latest["step"] == 42
    assert latest["trainer"] == {"next_ep_id": 7}
    assert (tmp_path / "checkpoints" / "step_000000042.pt").is_file()
    buffer.save.assert_called_once_with(tmp_path / "latest_buffer")


def test_best_checkpoint_uses_same_payload(tmp_path: Any) -> None:
    manager = CheckpointManager(tmp_path, MagicMock(), checkpoint_buffer=False)
    manager.save_best(_Agent(), 8, {"best_success": 0.5}, success=0.75)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["step"] == 8
    assert best["trainer"]["best_success"] == 0.5
