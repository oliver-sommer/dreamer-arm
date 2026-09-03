"""Checkpoint and replay-buffer persistence for online training."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)


class CheckpointManager:
    """Write latest, archived, and best checkpoints with a stable payload."""

    def __init__(self, root: str | Path, buffer: Any, checkpoint_buffer: bool) -> None:
        self._root = Path(root)
        self._buffer = buffer
        self._checkpoint_buffer = checkpoint_buffer

    @staticmethod
    def payload(agent: Any, env_step: int, trainer_state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "agent": agent.checkpoint_state(),
            "step": env_step,
            "trainer": dict(trainer_state),
        }

    def save(
        self,
        agent: Any,
        env_step: int,
        trainer_state: Mapping[str, Any],
        *,
        archive: bool = False,
    ) -> None:
        latest = self._root / "latest.pt"
        self._write(self.payload(agent, env_step, trainer_state), latest)
        log.info("checkpoint saved at step %d → %s", env_step, latest)
        if archive:
            kept = self._root / "checkpoints" / f"step_{env_step:09d}.pt"
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(latest, kept)
            log.info("archived checkpoint → %s", kept)
        if self._checkpoint_buffer:
            self._save_buffer(latest)

    def save_best(
        self,
        agent: Any,
        env_step: int,
        trainer_state: Mapping[str, Any],
        success: float,
    ) -> None:
        path = self._root / "best.pt"
        self._write(self.payload(agent, env_step, trainer_state), path)
        log.info("new best eval_robust/success_mean %.4f at step %d → %s", success, env_step, path)

    @staticmethod
    def _write(payload: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _save_buffer(self, latest: Path) -> None:
        target = latest.with_name(f"{latest.stem}_buffer")
        try:
            self._buffer.save(target)
        except Exception:
            log.exception("replay buffer save failed (%s); continuing without it", target)
