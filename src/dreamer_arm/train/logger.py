"""Weights & Biases logger for the Dreamer trainer.

Replaces the reference repo's TensorBoard + JSONL + stdout mix with a single
W&B-only sink. The logger buffers metrics between :meth:`write` calls so the
trainer can accumulate scalars / videos / histograms throughout a step and
flush them in one ``wandb.log`` call.

Pass ``mode="disabled"`` (or set it in the config) to suppress W&B for CI /
local dry-runs.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import wandb
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("wandb is required for dreamer_arm.train.logger; install via pixi.") from exc


_Scalar = float | int
_Array = np.ndarray | torch.Tensor


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_uint8_video(frames: _Array) -> np.ndarray:
    """Normalise a ``(B?, T, H, W, C)`` array into ``(T, C, H, B*W)`` uint8.

    Accepts an optional leading batch axis. Batched videos are tiled
    horizontally so a single W&B video panel shows all rollouts side-by-side
    — the same trick the reference repo used for its TensorBoard panel.
    """
    arr = _to_numpy(frames)
    if arr.ndim == 4:
        arr = arr[None]  # add batch axis
    if arr.ndim != 5:
        raise ValueError(f"expected video shape (B, T, H, W, C) or (T, H, W, C); got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    b, t, h, w, c = arr.shape
    # (B, T, H, W, C) -> (T, H, B, W, C) -> (T, C, H, B*W)
    arr = arr.transpose(1, 4, 2, 0, 3).reshape(t, c, h, b * w)
    return arr


class WandbLogger:
    """Buffered W&B logger.

    Usage::

        logger = WandbLogger(project="dreamer-arm", config=cfg, name="yam-r2dreamer-0")
        logger.scalar("loss/world_model", 1.23)
        logger.video("rollout/pred", frames)  # (B, T, H, W, C)
        logger.write(step=1000, fps=True)
        ...
        logger.finish()
    """

    def __init__(
        self,
        project: str,
        config: Mapping[str, Any] | None = None,
        name: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        mode: str | None = None,
        logdir: str | Path | None = None,
        video_fps: int = 16,
    ) -> None:
        if mode is None:
            mode = "online"
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            mode=mode,  # type: ignore[arg-type]
            config=dict(config) if config is not None else None,
            dir=str(logdir) if logdir is not None else None,
            reinit=True,
        )
        self._video_fps = video_fps
        self._scalars: dict[str, _Scalar] = {}
        self._videos: dict[str, np.ndarray] = {}
        self._images: dict[str, np.ndarray] = {}
        self._histograms: dict[str, np.ndarray] = {}
        self._last_step: int | None = None
        self._last_time: float | None = None

    # --------------------------------------------------------------- buffering

    def scalar(self, name: str, value: _Scalar | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        self._scalars[name] = float(value)

    def scalars(self, values: Mapping[str, _Scalar | torch.Tensor]) -> None:
        for k, v in values.items():
            self.scalar(k, v)

    def video(self, name: str, frames: _Array) -> None:
        self._videos[name] = _as_uint8_video(frames)

    def image(self, name: str, image: _Array) -> None:
        self._images[name] = _to_numpy(image)

    def histogram(self, name: str, values: _Array) -> None:
        self._histograms[name] = _to_numpy(values)

    # ------------------------------------------------------------------- flush

    def write(self, step: int, fps: bool = False) -> None:
        fps_value = self._compute_fps(step) if fps else None
        print(self._console_line(step, fps_value), flush=True)
        payload: dict[str, Any] = dict(self._scalars)
        if fps_value is not None:
            payload["fps/fps"] = fps_value
        for name, arr in self._videos.items():
            payload[name] = wandb.Video(arr, fps=self._video_fps, format="mp4")
        for name, arr in self._images.items():
            payload[name] = wandb.Image(arr)
        for name, arr in self._histograms.items():
            payload[name] = wandb.Histogram(arr.tolist())
        if payload:
            wandb.log(payload, step=step)
        self._scalars.clear()
        self._videos.clear()
        self._images.clear()
        self._histograms.clear()

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None  # type: ignore[assignment]

    # --------------------------------------------------------------- internals

    def _console_line(self, step: int, fps_value: float | None) -> str:
        # --- header: step + optional fps / episode scores ---
        parts = [f"step {step:>8}"]
        if fps_value is not None:
            parts.append(f"fps {fps_value:>6.1f}")
        for key, label in (("episode/score", "score"), ("episode/eval_score", "eval")):
            if key in self._scalars:
                parts.append(f"{label} {self._scalars[key]:.2f}")
        header = "  ".join(parts)

        # --- loss sub-line: only present on train-burst flushes ---
        loss_keys = [
            ("train/dyn", "dyn"),
            ("train/rep", "rep"),
            ("train/rew", "rew"),
            ("train/con", "con"),
            ("train/policy", "policy"),
            ("train/value", "value"),
        ]
        loss_parts = [
            f"{label} {self._scalars[key]:.3f}" for key, label in loss_keys if key in self._scalars
        ]
        return header + ("\n  " + "  ".join(loss_parts) if loss_parts else "")

    def _compute_fps(self, step: int) -> float:
        now = time.time()
        if self._last_step is None or self._last_time is None:
            self._last_step, self._last_time = step, now
            return 0.0
        steps = step - self._last_step
        elapsed = now - self._last_time
        self._last_step, self._last_time = step, now
        return steps / elapsed if elapsed > 0 else 0.0

    # ----------------------------------------------------------- context mgr

    def __enter__(self) -> WandbLogger:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.finish()
