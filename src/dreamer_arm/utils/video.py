"""Video layout conversion and ffmpeg encoding."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import imageio
import numpy as np
import torch
import wandb

_Array = np.ndarray | torch.Tensor


class VideoEncodingUnavailable(RuntimeError):
    """Raised when imageio cannot find a usable ffmpeg binary."""


def as_uint8_video(frames: _Array, cols: int | None = None) -> np.ndarray:
    """Convert B,T,H,W,C or T,H,W,C frames to a tiled T,C,H,W uint8 array."""
    arr = frames.detach().cpu().numpy() if isinstance(frames, torch.Tensor) else np.asarray(frames)
    if arr.ndim == 4:
        arr = arr[None]
    if arr.ndim != 5:
        raise ValueError(f"expected video shape (B, T, H, W, C) or (T, H, W, C); got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    batch, time, height, width, channels = arr.shape
    n_cols = cols if cols is not None else batch
    if n_cols <= 0 or batch % n_cols:
        raise ValueError(f"video batch size {batch} must be divisible by cols={n_cols}")
    n_rows = batch // n_cols
    arr = arr.reshape(n_rows, n_cols, time, height, width, channels)
    return arr.transpose(2, 5, 0, 3, 1, 4).reshape(time, channels, n_rows * height, n_cols * width)


def encode_video(arr: np.ndarray, fps: int) -> wandb.Video:
    """Encode T,C,H,W uint8 frames to a temporary MP4 for asynchronous upload."""
    frames = arr.transpose(0, 2, 3, 1)
    descriptor, temporary = tempfile.mkstemp(suffix=".mp4")
    os.close(descriptor)
    try:
        imageio.mimwrite(temporary, list(frames), fps=fps)
    except RuntimeError as exc:
        if "ffmpeg" not in str(exc).lower():
            raise
        Path(temporary).unlink()
        raise VideoEncodingUnavailable(str(exc)) from exc
    return wandb.Video(temporary, format="mp4")
