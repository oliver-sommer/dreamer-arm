"""Config discovery, composition and validation.

These are the cheapest tests in the suite and cover the most CLI surface: a
broken default, a stale interpolation or a mis-declared ``# @package`` header
would otherwise only surface minutes into a real run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamer_arm.utils.config import get_config_root, validate_config

CONFIG_ROOT = get_config_root()

#: Every entrypoint config, with the minimum overrides needed to satisfy ``???``.
ENTRYPOINTS = [
    ("training/dreamer", ["envs.task=MT10"]),
    ("inference/evaluate", ["checkpoint=/tmp/ckpt.pt", "envs.task=MT10"]),
    ("training/dreamer", ["core/model=dinowm", "envs.task=MT10"]),
    ("inference/evaluate", ["core/model=dinowm", "checkpoint=/tmp/ckpt.pt", "envs.task=MT10"]),
]


def _compose(config_name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def test_config_root_exists() -> None:
    assert CONFIG_ROOT.is_dir()
    assert (CONFIG_ROOT / "training" / "dreamer.yaml").is_file()


def test_config_tree_mirrors_package() -> None:
    """Every config group must correspond to a package under ``src/dreamer_arm``."""
    package_root = Path(__file__).resolve().parents[2] / "src" / "dreamer_arm"
    groups = {p.name for p in CONFIG_ROOT.iterdir() if p.is_dir()}
    packages = {p.name for p in package_root.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    assert groups <= packages, f"config groups with no matching package: {groups - packages}"


@pytest.mark.parametrize(("config_name", "overrides"), ENTRYPOINTS)
def test_entrypoint_composes_and_resolves(config_name: str, overrides: list[str]) -> None:
    cfg = _compose(config_name, overrides)
    validate_config(cfg)
    # Resolving proves every ${...} interpolation still points somewhere real.
    OmegaConf.resolve(cfg)
    assert cfg.entrypoint._target_.startswith("dreamer_arm.")


@pytest.mark.parametrize(("config_name", "overrides"), ENTRYPOINTS)
def test_entrypoint_target_is_importable(config_name: str, overrides: list[str]) -> None:
    import importlib

    target = _compose(config_name, overrides).entrypoint._target_
    module_name, _, attr = target.rpartition(".")
    assert hasattr(importlib.import_module(module_name), attr), f"{target} does not exist"


def test_composed_config_mirrors_package_layout() -> None:
    """``core/model`` must land at ``cfg.core.model``, not ``cfg.model``."""
    cfg = _compose("training/dreamer", ["envs.task=MT10"])
    assert "model" in cfg.core
    assert "buffer" in cfg.core
    assert "arm" in cfg.envs
    assert "model" not in cfg, "core/model leaked to the config root"


def test_env_and_arm_groups_are_selectable() -> None:
    cfg = _compose("training/dreamer", ["envs=metaworld", "envs.task=door-open", "envs/arm=sawyer"])
    assert cfg.envs.name == "metaworld"
    assert cfg.envs.task == "door-open"
    assert cfg.envs.arm.name == "sawyer"


def test_model_group_is_selectable() -> None:
    cfg = _compose("training/dreamer", ["core/model=dreamerv3", "envs.task=MT10"])
    assert cfg.core.model.rep_loss == "dreamerv3"


def test_dinowm_selectable_via_core_model_alone() -> None:
    """`core/model=dinowm` alone (no separate `training/dreamer-dinowm.yaml`
    recipe, no separate `core/buffer=dinowm`) must set `wm`, bump `envs.size`
    to match the ViT patch grid, and have the buffer's `max_size` follow it
    automatically (computed from `envs.size` -- see `core/buffer.yaml`) --
    exactly the same pattern `core/model=dreamerv3` already uses for the RSSM
    variants.
    """
    cfg = _compose("training/dreamer", ["core/model=dinowm", "envs.task=MT10"])
    assert cfg.core.model.wm == "dinowm"
    assert cfg.core.buffer.max_size == 125000  # 4x fewer slots than the RSSM default: 4x the bytes/frame at 128x128
    assert list(cfg.envs.size) == [128, 128]
    validate_config(cfg)
    OmegaConf.resolve(cfg)


def test_dinowm_selectable_for_eval_without_a_buffer() -> None:
    """Eval never builds a replay buffer (`inference/evaluate.yaml` doesn't
    include `/core/buffer` at all), so `core/model=dinowm` alone must compose
    cleanly -- the model config's `envs.size` override must not accidentally
    require a buffer node to exist.
    """
    cfg = _compose("inference/evaluate", ["core/model=dinowm", "checkpoint=/tmp/ckpt.pt", "envs.task=MT10"])
    assert cfg.core.model.wm == "dinowm"
    assert list(cfg.envs.size) == [128, 128]
    assert "buffer" not in cfg.core
    validate_config(cfg)
    OmegaConf.resolve(cfg)


def test_dinowm_requires_imag_starts() -> None:
    cfg = _compose(
        "training/dreamer",
        ["core/model=dinowm", "envs.task=MT10", "core.model.imag_starts=null"],
    )
    with pytest.raises(ValueError, match="requires core.model.imag_starts"):
        validate_config(cfg)


def test_dinowm_rejects_size_not_divisible_by_patch_size() -> None:
    cfg = _compose(
        "training/dreamer",
        ["core/model=dinowm", "envs.task=MT10", "envs.size=[100,100]"],
    )
    with pytest.raises(ValueError, match="divisible by 16"):
        validate_config(cfg)


def test_bare_train_defaults_to_mt10() -> None:
    """No overrides = the multi-task default; nothing is left unset."""
    cfg = _compose("training/dreamer", [])
    assert cfg.envs.task == "MT10"
    validate_config(cfg)


def test_single_task_group_requires_a_task() -> None:
    """`envs=metaworld` leaves envs.task=??? and must fail loudly."""
    cfg = _compose("training/dreamer", ["envs=metaworld"])
    with pytest.raises(ValueError, match="Missing required configuration"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("seed=-1", "seed must be non-negative"),
        ("envs.env_num=0", "envs.env_num must be positive"),
        ("envs.action_repeat=0", "envs.action_repeat must be positive"),
        ("core.buffer.max_size=10", "buffer.max_size"),
        ("trainer.train_ratio=-1", "trainer.train_ratio must be non-negative"),
        ("logging.wandb.mode=sometimes", "Unsupported logging.wandb.mode"),
    ],
)
def test_validate_config_rejects(override: str, match: str) -> None:
    cfg = _compose("training/dreamer", ["envs.task=MT10", override])
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_validate_config_accepts_the_defaults() -> None:
    validate_config(_compose("training/dreamer", ["envs.task=MT10"]))
