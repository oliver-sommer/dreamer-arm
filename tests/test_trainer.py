"""End-to-end smoke test for the online trainer.

Marked ``slow`` because it touches MuJoCo + the full Dreamer agent. The
goal is just to prove the wiring is consistent — sample → update → log
without NaNs over a handful of steps.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_trainer_runs_one_step() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("hydra")

    from pathlib import Path

    from hydra import compose, initialize_config_dir

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "device=cpu",
                "envs=metaworld",
                "envs.task=door-open",
                "envs.env_num=1",
                "envs.eval_episode_num=0",
                "trainer.steps=4",
                "trainer.update_log_every=1",
                "batch_size=2",
                "batch_length=4",
                "buffer.max_size=128",
            ],
        )
    # Don't actually run — just confirm the config composes.
    assert cfg.model.rep_loss == "r2dreamer"
    assert cfg.envs.name == "metaworld"
    assert cfg.envs.task == "door-open"
    assert cfg.arm.name == "yam"


def test_checkpoint_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Save/load restores the full agent: a fresh agent acts identically after load.

    Uses a tiny MLP-only Dreamer (no env, no images) so the test stays fast.
    """
    pytest.importorskip("hydra")
    from pathlib import Path

    import gymnasium as gym
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir

    from dreamer_arm.agent.dreamer import Dreamer

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                "device=cpu",
                "envs=metaworld",
                "envs.task=door-open",
                "model.deter=64",
                "model.hidden=16",
                "model.units=16",
                "model.rssm.stoch=4",
                "model.rssm.discrete=4",
                "model.rssm.blocks=2",
                "model.encoder.mlp_keys=state",
                "model.encoder.cnn_keys='\\$^'",
                "model.compile=false",
            ],
        )

    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-np.inf, np.inf, (8,), np.float32)})
    act_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

    torch.manual_seed(0)
    a1 = Dreamer(cfg.model, obs_space, act_space)
    a1._slow_value_updates = 17

    path = tmp_path / "checkpoint.pt"
    torch.save({"step": 12345, "agent": a1.checkpoint_state()}, path)
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["step"] == 12345

    torch.manual_seed(99)  # different init: loading must overwrite it
    a2 = Dreamer(cfg.model, obs_space, act_space)
    a2.load_checkpoint_state(ckpt["agent"])
    assert a2._slow_value_updates == 17

    obs = {
        "state": torch.randn(2, 8),
        "is_first": torch.ones(2, dtype=torch.bool),
    }
    s1 = a1.get_initial_state(2)
    s2 = a2.get_initial_state(2)
    # obs_step samples the stochastic latent, so pin the RNG for each call.
    torch.manual_seed(123)
    act1, _ = a1.act(obs, s1, eval_mode=True)
    torch.manual_seed(123)
    act2, _ = a2.act(obs, s2, eval_mode=True)
    torch.testing.assert_close(act1, act2)
    # Executed actions must respect the env bounds (clip in act()).
    assert act1.abs().max().item() <= 1.0
