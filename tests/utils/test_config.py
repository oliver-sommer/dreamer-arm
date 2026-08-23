"""Config discovery, composition and validation.

These are the cheapest tests in the suite and cover the most CLI surface: a
broken default, a stale interpolation or a mis-declared ``# @package`` header
would otherwise only surface minutes into a real run.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamer_arm.utils.config import get_config_root, run_hydra, validate_config

CONFIG_ROOT = get_config_root()

#: Every entrypoint config, with the minimum overrides needed to satisfy ``???``.
ENTRYPOINTS = [
    ("training/dreamer", ["envs.sim.task=MT10"]),
    ("inference/evaluate", ["checkpoint=/tmp/ckpt.pt", "envs.sim.task=MT10"]),
    ("training/dreamer", ["core/model=dinowm", "envs.sim.task=MT10"]),
    ("inference/evaluate", ["core/model=dinowm", "checkpoint=/tmp/ckpt.pt", "envs.sim.task=MT10"]),
    ("envs/sim/controller_bench", []),
]


def _compose(config_name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def test_config_root_exists() -> None:
    assert CONFIG_ROOT.is_dir()
    assert (CONFIG_ROOT / "training" / "dreamer.yaml").is_file()


def test_config_tree_mirrors_package() -> None:
    """Config groups and test packages must mirror source packages."""
    package_root = Path(__file__).resolve().parents[2] / "src" / "dreamer_arm"
    test_root = Path(__file__).resolve().parents[1]

    config_groups = {
        path.relative_to(CONFIG_ROOT)
        for path in CONFIG_ROOT.rglob("*")
        if path.is_dir() and not path.name.startswith(".")
    }
    unmatched_groups = {
        path
        for path in config_groups
        if not (package_root / path / "__init__.py").is_file()
        and not (package_root / path).with_suffix(".py").is_file()
    }
    assert not unmatched_groups, f"config groups with no matching source package or module: {unmatched_groups}"

    source_packages = {
        path.parent.relative_to(package_root)
        for path in package_root.rglob("__init__.py")
        if path.parent != package_root
    }
    test_packages = {
        path.parent.relative_to(test_root) for path in test_root.rglob("__init__.py") if path.parent != test_root
    }
    assert test_packages == source_packages, (
        f"tests missing packages: {source_packages - test_packages}; "
        f"tests with no source package: {test_packages - source_packages}"
    )

    test_modules = set()
    for path in test_root.rglob("test_*.py"):
        relative_path = path.relative_to(test_root)
        source_name = "__init__.py" if path.stem == "test_init" else path.stem.removeprefix("test_") + ".py"
        test_modules.add(relative_path.with_name(source_name))
    unmatched_tests = {path for path in test_modules if not (package_root / path).is_file()}
    assert not unmatched_tests, f"tests with no matching source module: {unmatched_tests}"


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


def test_run_hydra_help_composes_without_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task = MagicMock()
    monkeypatch.setattr("sys.argv", ["dreamer-arm-train", "--help", "envs.sim.task=MT10"])
    with pytest.raises(SystemExit) as exc_info:
        run_hydra(task, config_name="training/dreamer")
    assert exc_info.value.code == 0
    assert "Resolved configuration" in capsys.readouterr().out
    task.assert_not_called()


def test_composed_config_mirrors_package_layout() -> None:
    """``core/model`` must land at ``cfg.core.model``, not ``cfg.model``."""
    cfg = _compose("training/dreamer", ["envs.sim.task=MT10"])
    assert "model" in cfg.core
    assert "buffer" in cfg.core
    assert "sim" in cfg.envs
    assert "arms" in cfg.envs.sim
    assert "model" not in cfg, "core/model leaked to the config root"


def test_env_and_arm_groups_are_selectable() -> None:
    cfg = _compose(
        "training/dreamer",
        ["envs/sim=metaworld", "envs.sim.task=door-open", "envs/sim/arms=sawyer"],
    )
    assert cfg.envs.sim.name == "metaworld"
    assert cfg.envs.sim.task == "door-open"
    assert cfg.envs.sim.arms.name == "sawyer"


def test_model_group_is_selectable() -> None:
    cfg = _compose("training/dreamer", ["core/model=dreamerv3", "envs.sim.task=MT10"])
    assert cfg.core.model.rep_loss == "dreamerv3"


def test_dinowm_selectable_via_core_model_alone() -> None:
    """`core/model=dinowm` alone (no separate `training/dreamer-dinowm.yaml`
    recipe, no separate `core/buffer=dinowm`) must set `wm`, bump `envs.sim.size`
    to match the ViT patch grid, and have the buffer's `max_size` follow it
    automatically (computed from `envs.sim.size` -- see `core/buffer.yaml`) --
    exactly the same pattern `core/model=dreamerv3` already uses for the RSSM
    variants.
    """
    cfg = _compose("training/dreamer", ["core/model=dinowm", "envs.sim.task=MT10"])
    assert cfg.core.model.wm == "dinowm"
    assert cfg.core.buffer.max_size == 125000  # 4x fewer slots than the RSSM default: 4x the bytes/frame at 128x128
    assert list(cfg.envs.sim.size) == [128, 128]
    validate_config(cfg)
    OmegaConf.resolve(cfg)


def test_dinowm_selectable_for_eval_without_a_buffer() -> None:
    """Eval never builds a replay buffer (`inference/evaluate.yaml` doesn't
    include `/core/buffer` at all), so `core/model=dinowm` alone must compose
    cleanly -- the model config's `envs.sim.size` override must not accidentally
    require a buffer node to exist.
    """
    cfg = _compose(
        "inference/evaluate",
        ["core/model=dinowm", "checkpoint=/tmp/ckpt.pt", "envs.sim.task=MT10"],
    )
    assert cfg.core.model.wm == "dinowm"
    assert list(cfg.envs.sim.size) == [128, 128]
    assert "buffer" not in cfg.core
    validate_config(cfg)
    OmegaConf.resolve(cfg)


def test_dinowm_requires_imag_starts() -> None:
    cfg = _compose(
        "training/dreamer",
        ["core/model=dinowm", "envs.sim.task=MT10", "core.model.imag_starts=null"],
    )
    with pytest.raises(ValueError, match="requires core.model.imag_starts"):
        validate_config(cfg)


def test_dinowm_rejects_size_not_divisible_by_patch_size() -> None:
    cfg = _compose(
        "training/dreamer",
        ["core/model=dinowm", "envs.sim.task=MT10", "envs.sim.size=[100,100]"],
    )
    with pytest.raises(ValueError, match="divisible by 16"):
        validate_config(cfg)


def test_bare_train_defaults_to_mt10() -> None:
    """No overrides = the multi-task default; nothing is left unset."""
    cfg = _compose("training/dreamer", [])
    assert cfg.envs.sim.task == "MT10"
    validate_config(cfg)


def test_default_policy_rate_preserves_metaworld_horizon() -> None:
    cfg = _compose("training/dreamer", [])
    assert cfg.envs.sim.action_repeat == 1
    assert cfg.envs.sim.time_limit == 500
    assert cfg.envs.sim.action_repeat * cfg.envs.sim.time_limit == 500


def test_default_eval_warmup_schedule() -> None:
    cfg = _compose("training/dreamer", [])
    assert list(cfg.trainer.eval_warmup_steps) == [1000, 2500, 5000, 10000, 15000]


def test_single_task_group_requires_a_task() -> None:
    """`envs/sim=metaworld` leaves envs.sim.task=??? and must fail loudly."""
    cfg = _compose("training/dreamer", ["envs/sim=metaworld"])
    with pytest.raises(ValueError, match="Missing required configuration"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("seed=-1", "seed must be non-negative"),
        ("envs.sim.env_num=0", "envs.sim.env_num must be positive"),
        ("envs.sim.action_repeat=0", "envs.sim.action_repeat must be positive"),
        ("core.buffer.max_size=10", "buffer.max_size"),
        ("trainer.train_ratio=-1", "trainer.train_ratio must be non-negative"),
        ("trainer.eval_warmup_steps=[0,1000]", "eval_warmup_steps"),
        ("logging.wandb.mode=sometimes", "Unsupported logging.wandb.mode"),
    ],
)
def test_validate_config_rejects(override: str, match: str) -> None:
    cfg = _compose("training/dreamer", ["envs.sim.task=MT10", override])
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_validate_config_accepts_the_defaults() -> None:
    validate_config(_compose("training/dreamer", ["envs.sim.task=MT10"]))


def test_inductor_cache_dir_is_absolute() -> None:
    """TORCHINDUCTOR_CACHE_DIR must not be a relative path.

    Inductor stores the value verbatim and later shells out to g++ with cwd set
    to a scratch build dir (_inductor/cpp_builder.py builds with
    ``run_compile_cmd(..., cwd=_build_tmp_dir)``).  A relative cache dir then
    resolves against that scratch dir, and the compile fails with "No such file
    or directory" naming its own generated .cpp.  torch's own default cache dir
    is absolute for this reason.

    Only CUDA hosts compile at all (``auto_compile`` is
    ``torch.cuda.is_available()``), so a regression here is invisible on macOS
    and only explodes on the training box -- worth pinning in config.
    """
    import re
    import tomllib
    from pathlib import Path

    pixi_toml = Path(__file__).resolve().parents[2] / "pixi.toml"
    config = tomllib.loads(pixi_toml.read_text())

    found = 0
    for name, task in config.get("tasks", {}).items():
        if not isinstance(task, dict):
            continue
        value = task.get("env", {}).get("TORCHINDUCTOR_CACHE_DIR")
        if value is None:
            continue
        found += 1
        # Either a literal absolute path, or rooted at a pixi-expanded variable.
        assert value.startswith("/") or re.match(r"^\$[A-Z_]*(ROOT|DIR)\b", value), (
            f"task {name!r} sets a relative TORCHINDUCTOR_CACHE_DIR={value!r}; "
            "inductor compiles from a different cwd and will not find it"
        )

    assert found, "no task sets TORCHINDUCTOR_CACHE_DIR — did the tasks get renamed?"
