"""Weights & Biases metric sink (scalars, videos, histograms).

Replaces the reference repo's TensorBoard + JSONL + stdout mix with a single
W&B-only sink. The logger buffers metrics between :meth:`write` calls so the
trainer can accumulate scalars / videos / histograms throughout a step and
flush them in one ``wandb.log`` call.

Pass ``mode="disabled"`` (or set it in the config) to suppress W&B for CI /
local dry-runs.

Pass ``mode="offline"`` on hosts with unreliable egress: the logger then
uploads the run itself via periodic ``wandb sync --append`` from a background
thread (every ``sync_interval`` seconds, plus a final sync on ``finish``),
which survives network failures that permanently kill wandb's online stream.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch

from dreamer_arm.utils.logging import CONTINUATION_INDENT

try:
    import wandb
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("wandb is required for dreamer_arm.utils.tracking; install via pixi.") from exc

log = logging.getLogger(__name__)


_Scalar = float | int
_Array = np.ndarray | torch.Tensor


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_uint8_video(frames: _Array, cols: int | None = None) -> np.ndarray:
    """Normalise a ``(B?, T, H, W, C)`` array into ``(T, C, rows*H, cols*W)`` uint8.

    Accepts an optional leading batch axis. ``cols`` controls how many videos
    appear per row; the remainder fill the rows below. Defaults to all videos
    in a single row (``cols=B``).
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
    n_cols = cols if cols is not None else b
    n_rows = b // n_cols
    # (B, T, H, W, C) -> (rows, cols, T, H, W, C) -> (T, C, rows*H, cols*W)
    arr = arr.reshape(n_rows, n_cols, t, h, w, c)
    arr = arr.transpose(2, 5, 0, 3, 1, 4).reshape(t, c, n_rows * h, n_cols * w)
    return arr


class WandbLogger:
    """Buffered W&B logger.

    Usage::

        logger = WandbLogger(project="dreamer-arm", config=cfg)
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
        sync_interval: float = 600.0,
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
        self._max_logged_step: int | None = None
        self._last_log_time: float = time.time()
        self._keepalive_secs: float = 60.0

        # Offline mode + periodic `wandb sync --append` is the robust setup for
        # hosts with unreliable egress (e.g. vast.ai, whose proxy intermittently
        # MITMs TLS to api.wandb.ai; wandb-core treats certificate errors as
        # non-retryable and kills the online filestream *permanently*, leaving
        # the run "crashed" on the website while training continues).  Each
        # sync is a fresh subprocess, so a transient failure only delays the
        # upload by one interval instead of ending it.
        self._sync_stop = threading.Event()
        self._sync_thread: threading.Thread | None = None
        if mode == "offline" and sync_interval > 0:
            self._sync_thread = threading.Thread(target=self._sync_loop, args=(float(sync_interval),), daemon=True)
            self._sync_thread.start()

    # --------------------------------------------------------------- buffering

    def scalar(self, name: str, value: _Scalar | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        self._scalars[name] = float(value)

    def scalars(self, values: Mapping[str, _Scalar | torch.Tensor]) -> None:
        for k, v in values.items():
            self.scalar(k, v)

    def video(self, name: str, frames: _Array, cols: int | None = None) -> None:
        self._videos[name] = _as_uint8_video(frames, cols=cols)

    def image(self, name: str, image: _Array) -> None:
        self._images[name] = _to_numpy(image)

    def histogram(self, name: str, values: _Array) -> None:
        self._histograms[name] = _to_numpy(values)

    # ------------------------------------------------------------------- flush

    def write(self, step: int, fps: bool = False) -> None:
        # wandb silently drops any log whose step <= the previous committed step.
        # Nudge by 1 on collision so eval payloads (including video) always land.
        if self._max_logged_step is not None and step <= self._max_logged_step:
            step = self._max_logged_step + 1
        self._max_logged_step = step

        fps_value = self._compute_fps(step) if fps else None
        log.info(self._console_line(step, fps_value))
        payload: dict[str, Any] = dict(self._scalars)
        if fps_value is not None:
            payload["fps/fps"] = fps_value
        for name, arr in self._videos.items():
            payload[name] = self._encode_video(arr)
        for name, arr in self._images.items():
            payload[name] = wandb.Image(arr)
        for name, arr in self._histograms.items():
            payload[name] = wandb.Histogram(arr.tolist())
        now = time.time()
        if payload:
            wandb.log(payload, step=step)
            self._last_log_time = now
        elif now - self._last_log_time >= self._keepalive_secs:
            wandb.log({}, step=step)
            self._last_log_time = now
        self._scalars.clear()
        self._videos.clear()
        self._images.clear()
        self._histograms.clear()

    def keepalive(self, step: int) -> None:
        """Send an empty wandb.log if no log has been sent recently."""
        now = time.time()
        if now - self._last_log_time >= self._keepalive_secs:
            wandb.log({}, step=step)
            self._last_log_time = now

    def finish(self) -> None:
        if self._run is not None:
            sync_dir = str(self._run.settings.sync_dir)
            had_syncer = self._sync_thread is not None
            if self._sync_thread is not None:
                self._sync_stop.set()
                self._sync_thread.join(timeout=5.0)
                self._sync_thread = None
            self._run.finish()
            self._run = None  # type: ignore[assignment]
            if had_syncer:
                # Final sync after finish() so the run's end state is uploaded.
                self._sync_once(sync_dir)

    # --------------------------------------------------------------- internals

    def _sync_loop(self, interval: float) -> None:
        sync_dir = str(self._run.settings.sync_dir)
        while not self._sync_stop.wait(interval):
            self._sync_once(sync_dir)

    @staticmethod
    def _sync_once(sync_dir: str) -> None:
        """Upload the offline run directory; failures are logged, never raised."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "wandb", "sync", "--append", sync_dir],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if result.returncode != 0:
                tail = (result.stderr or result.stdout).strip().splitlines()[-1:]
                log.warning("wandb sync failed (will retry): %s", " ".join(tail))
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("wandb sync failed (will retry): %s", exc)

    def _encode_video(self, arr: np.ndarray) -> wandb.Video:
        """Encode (T, C, H, W) uint8 video via imageio-ffmpeg, then hand wandb a path.

        wandb.Video(numpy_array) calls ffmpeg as a subprocess in a way that can
        hang on headless Linux servers.  Writing the file ourselves with imageio
        (which ships its own ffmpeg binary) and passing the resulting path avoids
        that code path entirely.

        The temp file is left on disk because wandb.Video stores the path and
        uploads the file asynchronously; deleting it before the upload would
        silently drop the video.  OS temp-dir cleanup handles the rest.
        """
        # _as_uint8_video produces (T, C, H, W); imageio wants (T, H, W, C).
        frames = arr.transpose(0, 2, 3, 1)
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        imageio.mimwrite(tmp_path, list(frames), fps=self._video_fps)
        return wandb.Video(tmp_path, format="mp4")

    def _console_line(self, step: int, fps_value: float | None) -> str:
        # --- header: step + optional fps / episode scores ---
        # The logging formatter prepends the bracketed timestamp + level.
        parts = [f"step {step:>8}"]
        if fps_value is not None:
            parts.append(f"fps {fps_value:>6.1f}")
        for key, label in (("episode/score", "score"), ("episode/eval_score", "eval")):
            if key in self._scalars:
                parts.append(f"{label} {self._scalars[key]:.2f}")
        header = "  ".join(parts)

        # --- loss sub-line: only present on train-burst flushes ---
        loss_keys = [
            ("train/loss/dyn", "dyn"),
            ("train/loss/rep", "rep"),
            ("train/loss/rew", "rew"),
            ("train/loss/con", "con"),
            ("train/loss/policy", "policy"),
            ("train/loss/value", "value"),
        ]
        loss_parts = [f"{label} {self._scalars[key]:.3f}" for key, label in loss_keys if key in self._scalars]
        # Indent the continuation line flush under the message column (the
        # formatter prefix only stamps the first line).
        if loss_parts:
            return header + "\n" + CONTINUATION_INDENT + "  ".join(loss_parts)
        return header

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
