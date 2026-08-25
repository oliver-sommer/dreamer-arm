"""Hydra config discovery, validation and entrypoint dispatch.

Every command shares one shape: :func:`run_hydra` composes a config from
``configs/``, then hands it to :func:`dispatch`, which validates it and calls
whatever ``entrypoint._target_`` names.  A config therefore fully determines
which code runs, and adding a command means adding a config plus the module it
points at — no new argument parsing.

Config groups mirror the package layout: ``configs/core`` configures
:mod:`dreamer_arm.core`, ``configs/envs`` configures :mod:`dreamer_arm.envs`,
and so on.
"""

from __future__ import annotations

import logging
import os
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

from dreamer_arm.utils.logging import configure_logging

log = logging.getLogger(__name__)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _auto_compile() -> bool:
    return torch.cuda.is_available()


def _buffer_max_size(size: list[int] | tuple[int, int], budget_bytes: int) -> int:
    """Replay-buffer slot count that fits ``budget_bytes`` of uint8 RGB frames at ``size``.

    Keeps a fixed memory budget regardless of render resolution, so no
    per-world-model buffer variant is needed: this is the only thing
    ``core/buffer/dinowm.yaml`` used to exist to hand-recompute for 128x128
    frames instead of the RSSM default's 64x64.
    """
    h, w = size
    return budget_bytes // (int(h) * int(w) * 3)


OmegaConf.register_new_resolver("auto_device", _auto_device, replace=True)
OmegaConf.register_new_resolver("auto_compile", _auto_compile, replace=True)
OmegaConf.register_new_resolver("buffer_max_size", _buffer_max_size, replace=True)


def get_config_root() -> Path:
    """Locate ``configs/``, whether running from a checkout or an installed wheel."""
    candidates: list[Path] = []
    if override := os.environ.get("DREAMER_ARM_CONFIG_DIR"):
        candidates.append(Path(override))
    candidates.extend(
        [
            Path(__file__).resolve().parents[3] / "configs",
            Path(sysconfig.get_path("data")) / "share" / "dreamer-arm" / "configs",
        ]
    )
    for root in candidates:
        if root.is_dir():
            return root
    searched = ", ".join(map(str, candidates))
    raise FileNotFoundError(f"dreamer-arm Hydra config directory not found; searched: {searched}")


def validate_config(cfg: DictConfig) -> None:
    """Fail fast on configs that would only break minutes into a run.

    Catches the mistakes that are cheap to make from the CLI and expensive to
    discover later — a mandatory task left unset, a batch shape the buffer
    cannot serve, an eval cadence that would consume the whole run.
    """
    missing = sorted(OmegaConf.missing_keys(cfg))
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")

    if int(cfg.seed) < 0:
        raise ValueError("seed must be non-negative")

    envs_root = cfg.get("envs")
    envs = envs_root.get("sim") if envs_root is not None else None
    if envs is not None:
        if int(envs.env_num) <= 0:
            raise ValueError("envs.sim.env_num must be positive")
        if int(envs.action_repeat) <= 0:
            raise ValueError("envs.sim.action_repeat must be positive")
        if int(envs.time_limit) <= 0:
            raise ValueError("envs.sim.time_limit must be positive")

    core = cfg.get("core")
    model = core.get("model") if core is not None else None
    if model is not None:
        wm = str(model.get("wm", "rssm"))
        if wm not in ("rssm", "dinowm", "dreamer4"):
            raise ValueError(f"Unsupported core.model.wm: {wm}")
        if wm != "rssm" and model.get("imag_starts", None) is None:
            raise ValueError(f"core.model.wm={wm!r} requires core.model.imag_starts to be set (not null)")
        size = envs.get("size") if envs is not None else None
        if wm in ("dinowm", "dreamer4") and size is not None:
            size = [int(x) for x in size]
            if any(s % 16 != 0 for s in size):
                raise ValueError(f"core.model.wm={wm!r} needs envs.sim.size divisible by 16 (patch size), got {size}")

    buffer = core.get("buffer") if core is not None else None
    if buffer is not None:
        if int(buffer.batch_size) <= 0 or int(buffer.batch_length) <= 0:
            raise ValueError("buffer.batch_size and buffer.batch_length must be positive")
        # The buffer stores (batch_length + 1) steps per slice: one context
        # step to seed the RSSM plus the training window.
        needed = int(envs.env_num) * (int(buffer.batch_length) + 1) if envs is not None else 0
        if needed and int(buffer.max_size) < needed:
            raise ValueError(
                f"buffer.max_size ({buffer.max_size}) is below the minimum fill "
                f"envs.sim.env_num * (buffer.batch_length + 1) = {needed}; "
                "training would never start"
            )

    trainer = cfg.get("trainer")
    if trainer is not None:
        if float(trainer.replay_ratio) < 0:
            raise ValueError("trainer.replay_ratio must be non-negative")
        if int(trainer.eval_episode_num) > 0 and int(trainer.eval_every) <= 0:
            raise ValueError("trainer.eval_every must be positive when eval is enabled")
        warmup_steps = [int(step) for step in trainer.get("eval_warmup_steps", [])]
        if any(step <= 0 for step in warmup_steps):
            raise ValueError("trainer.eval_warmup_steps must contain only positive steps")

    logging_cfg = cfg.get("logging")
    wandb = logging_cfg.get("wandb") if logging_cfg is not None else None
    if wandb is not None and str(wandb.mode) not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported logging.wandb.mode: {wandb.mode}")


def dispatch(cfg: DictConfig) -> object:
    """Validate ``cfg``, freeze it, and call its ``entrypoint._target_``."""
    validate_config(cfg)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cast(object, call(cfg.entrypoint, cfg=cfg, _recursive_=False))


def run_hydra[ResultT](
    task: Callable[[DictConfig], ResultT],
    *,
    config_name: str,
    selector: tuple[str, str] | None = None,
) -> ResultT:
    """Compose a config from CLI overrides and run ``task`` against it.

    ``selector`` allows one override to choose the *config* rather than a value
    inside it: ``selector=("training", "training")`` turns ``training=dreamer``
    into ``config_name="training/dreamer"``.  It is consumed before Hydra sees
    the overrides, so it never reaches the composed config.
    """
    overrides = list(sys.argv[1:])
    show_help = any(arg in {"-h", "--help"} for arg in overrides)
    overrides = [arg for arg in overrides if arg not in {"-h", "--help"}]
    if selector is not None:
        field, group = selector
        prefix = f"{field}="
        matches = [(index, arg.removeprefix(prefix)) for index, arg in enumerate(overrides) if arg.startswith(prefix)]
        if len(matches) > 1:
            raise SystemExit(f"Specify {field}=<name> only once")
        if matches:
            index, name = matches[0]
            if not name or "/" in name or name.startswith("."):
                raise SystemExit(f"Invalid {field} selection: {name!r}")
            config_name = f"{group}/{name}"
            del overrides[index]

    with initialize_config_dir(config_dir=str(get_config_root()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    if show_help:
        print(f"Usage: {Path(sys.argv[0]).name} [key=value ...]")
        print("\nResolved configuration:\n")
        print(OmegaConf.to_yaml(cfg, resolve=True))
        raise SystemExit(0)

    configure_logging(str(cfg.logging.level))

    # Hydra's ``compose`` API (unlike ``@hydra.main``) creates no run directory,
    # so own it here and drop the resolved config in it -- that file is what
    # makes a finished run reproducible.
    logdir = Path(str(cfg.logdir))
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    log.info("run directory: %s", logdir)

    try:
        return task(cfg)
    except KeyboardInterrupt:
        log.warning("run interrupted")
        raise SystemExit(130) from None


__all__ = ["dispatch", "get_config_root", "run_hydra", "validate_config"]
