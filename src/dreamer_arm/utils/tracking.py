"""Weights & Biases metric sink (scalars, videos, histograms, tables).

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
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dreamer_arm.utils.logging import CONTINUATION_INDENT
from dreamer_arm.utils.video import VideoEncodingUnavailable, as_uint8_video, encode_video

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
            reinit="finish_previous",
        )
        # W&B's internal row counter must increase on every log call, but
        # train and eval can legitimately emit separate rows for the same
        # environment transition (for example, both at the 2.5k milestone).
        # Use an explicit semantic axis instead of falsifying the environment
        # step to dodge W&B's monotonically-increasing internal counter.
        self._run.define_metric("env_step", hidden=True)
        self._run.define_metric("*", step_metric="env_step")
        self._video_fps = video_fps
        self._warned_no_ffmpeg = False
        self._scalars: dict[str, _Scalar] = {}
        self._pending: dict[str, torch.Tensor] = {}
        self._videos: dict[str, np.ndarray] = {}
        self._images: dict[str, np.ndarray] = {}
        self._histograms: dict[str, np.ndarray] = {}
        self._tables: dict[str, tuple[list[str], list[list[Any]]]] = {}
        self._last_step: int | None = None
        self._last_time: float | None = None
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

    def scalar(self, name: str, value: _Scalar | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            value = float(value.detach().cpu().item())
        self._scalars[name] = float(value)

    def scalars(self, values: Mapping[str, _Scalar | torch.Tensor], defer: bool = False) -> None:
        """Log a batch of scalars.

        With ``defer=True`` (the update-loop path), tensor values are kept as
        tensors in ``_pending`` and only synced to host in :meth:`write` --
        see :meth:`_flush_pending` for why that matters. With ``defer=False``
        (the default), tensors are synced immediately: at most one CUDA sync
        per device, not one per key, so a handful of scalars logged outside
        the update loop still don't pay a sync per key.

        Grouped by device rather than stacked in one call: a metrics dict that
        mixes a stray CPU-default ``torch.tensor(x)`` in with the rest (an easy
        slip -- it happened once already, see optim/step.py's history) would
        make a single ungrouped stack crash with a device-mismatch error that
        only a CUDA/MPS host would ever hit, since a CPU-only dev box has
        nothing to mismatch against.
        """
        tensors = {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}
        if defer:
            self._pending.update({k: v.detach() for k, v in tensors.items()})
        else:
            by_device: dict[torch.device, list[str]] = {}
            for k, v in tensors.items():
                by_device.setdefault(v.device, []).append(k)
            for _device, keys in by_device.items():
                # Every value here is a loss/metric mean -> single-element; reshape
                # rather than stacking mismatched shapes if that invariant ever slips.
                stacked = torch.stack([tensors[k].detach().reshape(()) for k in keys])
                for k, val in zip(keys, stacked.tolist(), strict=True):
                    self._scalars[k] = float(val)
        for k, v in values.items():
            if k not in tensors:
                self.scalar(k, v)

    def _flush_pending(self) -> None:
        """Sync every deferred tensor scalar to host, once, grouped by device.

        The update loop calls :meth:`scalars` with ``defer=True`` after every
        ``agent.update`` -- but only the last update's metrics before the next
        :meth:`write` are ever read (each call overwrites the last). Syncing
        immediately, as ``scalars(defer=False)`` does, would pay a blocking
        device->host stall after every update just to throw away all but one
        of them. Deferring collapses that to one sync per ``write``, no matter
        how many updates ran in between -- the same "one sync, not N" argument
        ``scalars`` used to make per-call, now amortised across the window.
        """
        if not self._pending:
            return
        by_device: dict[torch.device, list[str]] = {}
        for k, v in self._pending.items():
            by_device.setdefault(v.device, []).append(k)
        for _device, keys in by_device.items():
            stacked = torch.stack([self._pending[k].reshape(()) for k in keys])
            for k, val in zip(keys, stacked.tolist(), strict=True):
                self._scalars[k] = float(val)
        self._pending.clear()

    def video(self, name: str, frames: _Array, cols: int | None = None) -> None:
        self._videos[name] = as_uint8_video(frames, cols=cols)

    def image(self, name: str, image: _Array) -> None:
        self._images[name] = _to_numpy(image)

    def histogram(self, name: str, values: _Array) -> None:
        self._histograms[name] = _to_numpy(values)

    def table(self, name: str, columns: list[str], rows: list[list[Any]]) -> None:
        """Buffer structured rows as one W&B table instead of N scalar keys."""
        self._tables[name] = (list(columns), [list(row) for row in rows])

    def write(self, step: int, fps: bool = False) -> None:
        self._flush_pending()

        fps_value = self._compute_fps(step) if fps else None
        log.info(self._console_line(step))
        payload: dict[str, Any] = {"env_step": step, **self._scalars}
        if fps_value is not None:
            payload["fps/fps"] = fps_value
        for name, arr in self._videos.items():
            video = self._encode_video(arr)
            if video is not None:
                payload[name] = video
        for name, arr in self._images.items():
            payload[name] = wandb.Image(arr)
        for name, arr in self._histograms.items():
            payload[name] = wandb.Histogram(arr.tolist())
        for name, (columns, rows) in self._tables.items():
            payload[name] = wandb.Table(columns=columns, data=rows)
        now = time.time()
        # Do not pass W&B's reserved ``step=`` argument here. Its internal
        # history row remains monotonic while ``env_step`` can correctly be
        # equal across separate train/eval records.
        if len(payload) > 1 or now - self._last_log_time >= self._keepalive_secs:
            wandb.log(payload)
            self._last_log_time = now
        self._scalars.clear()
        self._videos.clear()
        self._images.clear()
        self._histograms.clear()
        self._tables.clear()

    def keepalive(self, step: int) -> None:
        """Send an empty wandb.log if no log has been sent recently."""
        now = time.time()
        if now - self._last_log_time >= self._keepalive_secs:
            wandb.log({"env_step": step})
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

    def _encode_video(self, arr: np.ndarray) -> wandb.Video | None:
        """Encode (T, C, H, W) uint8 video via imageio-ffmpeg, then hand wandb a path.

        wandb.Video(numpy_array) calls ffmpeg as a subprocess in a way that can
        hang on headless Linux servers.  Writing the file ourselves with imageio
        (which ships its own ffmpeg binary) and passing the resulting path avoids
        that code path entirely.

        The temp file is left on disk because wandb.Video stores the path and
        uploads the file asynchronously; deleting it before the upload would
        silently drop the video.  OS temp-dir cleanup handles the rest.

        Returns ``None`` (dropping the video, logging a one-time warning) if no
        ffmpeg binary is available -- e.g. imageio-ffmpeg's platform wheel failed
        to ship one, or the pixi env is out of sync -- so a broken/incomplete
        environment loses video logging instead of taking down the whole run.
        """
        try:
            return encode_video(arr, self._video_fps)
        except VideoEncodingUnavailable as exc:
            if not self._warned_no_ffmpeg:
                log.warning("Skipping video logging: %s", exc)
                self._warned_no_ffmpeg = True
            return None

    def _console_line(self, step: int) -> str:
        # The logging formatter prepends the bracketed timestamp + level.
        parts = [f"step {step:>8}"]
        for key, label in (("episode/score", "score"), ("episode/eval_score", "eval")):
            if key in self._scalars:
                parts.append(f"{label} {self._scalars[key]:.2f}")
        header = "  ".join(parts)

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

    def __enter__(self) -> WandbLogger:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.finish()
