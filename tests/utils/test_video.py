from __future__ import annotations

import numpy as np
import pytest
import torch

from dreamer_arm.utils.video import as_uint8_video


def test_as_uint8_video_tiles_batch() -> None:
    frames = torch.zeros((4, 3, 2, 5, 3), dtype=torch.float32)
    frames[1] = 1.0
    video = as_uint8_video(frames, cols=2)
    assert video.shape == (3, 3, 4, 10)
    assert video.dtype == np.uint8
    assert video[:, :, :2, 5:].max() == 255


def test_as_uint8_video_accepts_unbatched_frames() -> None:
    frames = np.zeros((3, 2, 5, 3), dtype=np.uint8)
    assert as_uint8_video(frames).shape == (3, 3, 2, 5)


@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 2, 3, 4, 5, 6)])
def test_as_uint8_video_rejects_invalid_rank(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="expected video shape"):
        as_uint8_video(np.zeros(shape))


def test_as_uint8_video_rejects_ragged_grid() -> None:
    with pytest.raises(ValueError, match="divisible"):
        as_uint8_video(np.zeros((3, 2, 4, 4, 3), dtype=np.uint8), cols=2)
