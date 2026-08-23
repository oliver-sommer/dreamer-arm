import random

import numpy as np
import torch

from dreamer_arm.utils.seed import set_seed_everywhere


def test_set_seed_everywhere_replays_all_rngs() -> None:
    set_seed_everywhere(17)
    first = (random.random(), np.random.random(), torch.rand(1))
    set_seed_everywhere(17)
    second = (random.random(), np.random.random(), torch.rand(1))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
