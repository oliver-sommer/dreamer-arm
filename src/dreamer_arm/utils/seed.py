"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed_everywhere(seed: int) -> None:
    """Seed Python's ``random``, NumPy, and PyTorch (CPU + all CUDA devices)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_deterministic_run() -> None:
    """Pin cuBLAS / cuDNN to deterministic kernels at the cost of throughput."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
