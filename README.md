# dreamer-arm

A Dreamer (**R2-Dreamer** / **DreamerV3**) implementation for the
[**i2rt YAM**][i2rt_yam] bimanual robotic arm, trained in MuJoCo via the
[DeepMind Control Suite][dmc] stack.

The world model is configurable behind a single config switch:

- `core/model=r2dreamer` *(default)* — decoder-free, augmentation-free. Trains
  the RSSM with a Barlow-Twins redundancy-reduction loss between the image encoder
  embedding and a projector over the latent state (per [Morihira et al., 2026][r2dreamer_paper]).
- `core/model=dreamerv3` — classic decoder-based reconstruction baseline.

Everything else (RSSM with categorical stochastic state, KL balancing,
actor–critic with λ-returns, LaProp + adaptive gradient clipping) is shared.

## Quick start

```bash
# install (init submodule first; TODO opt-A: drop this line once metaworld is a git URL dep)
git submodule update --init --recursive
pixi install

# train R2-Dreamer on multi-task Meta-World MT10 (default)
pixi run training

# larger multi-task benchmark, one env per task
pixi run training envs.task=MT50 envs.env_num=50

# single task, and with the Sawyer arm instead of YAM
pixi run training envs=metaworld envs.task=door-open
pixi run training envs=metaworld envs.task=door-open envs/arm=sawyer

# DreamerV3 baseline
pixi run training core/model=dreamerv3

# evaluate a saved checkpoint
pixi run inference checkpoint=logs/<date>/<time>/best.pt envs.task=MT10

# inspect Cartesian-controller tracking without launching training
pixi run controller-bench
pixi run controller-bench mode=servo
```

All runs log to Weights & Biases. Set `WANDB_API_KEY` (and optionally
`WANDB_ENTITY` / `WANDB_PROJECT`) before training; if unset, the logger
automatically falls back to `mode=disabled`. Pass `logging.wandb.mode=offline`
on hosts with unreliable egress.

## Project layout

`configs/` and `tests/` mirror the package layout, so a config group, its code
and its tests always sit at the same path.

```
src/dreamer_arm/
  core/               model + algorithm: agent, world models, networks, losses, buffer, optim
  controller/         Cartesian IK, runtime metrics, and controller bench
  envs/               MuJoCo / Meta-World envs, arm plugins, rendering, wrappers
  training/           online loop, checkpoint persistence, and composition root
  inference/          checkpoint evaluation (also used by the in-loop eval)
  utils/              config dispatch, console logging, W&B tracking, video, seeding
configs/              controller/  core/{model,buffer}/  envs/{,arm/}  training/  inference/  utils/logging/
tests/                mirrors src/dreamer_arm/ (no GPU required)
thirdparty/metaworld/ YAMetaworld submodule: YAM MJCF, meshes and task suite
logs/<date>/<time>/    latest.pt, best.pt, checkpoints/, videos/, config.yaml
```

Each command is one config: `configs/training/dreamer.yaml`,
`configs/inference/evaluate.yaml`, and `configs/controller/bench.yaml`
name their own `entrypoint._target_`, which `dreamer_arm.utils.config.dispatch`
calls after validating the config. Adding a command means adding a config plus
the module it points at.

```bash
python -m dreamer_arm.training                # = pixi run training
python -m dreamer_arm.inference.evaluate      # = pixi run inference
python -m dreamer_arm.controller.bench       # = pixi run controller-bench
```

## Development

```bash
pixi run -e dev hooks    # one-time: install the prek git hooks

pixi run -e dev check    # format-check + lint + typecheck + test (what CI runs)

pixi run -e dev format
pixi run -e dev lint       # ruff
pixi run -e dev typecheck  # ty
pixi run -e dev test       # pytest
```

When running an ad-hoc Python command or pytest invocation that imports both
PyTorch and MuJoCo, set `KMP_DUPLICATE_LIB_OK=TRUE`; their bundled OpenMP
runtimes otherwise conflict on macOS. The built-in Pixi tasks already set it.

## Template

This project is generated from [pytorch-template][template] and tracked with
[Copier][copier], so tooling changes made in the template can be pulled in
later. The answers live in `configs/.copier-template.yaml`.

```bash
pixi run -e dev template-update    # pull template changes (needs a clean worktree)
```

Copier applies only what changed between the recorded template version and the
new one, three-way merging into local edits, so project-specific customisations
in `pyproject.toml`, `pixi.toml` and `prek.toml` survive an update.

[template]: https://github.com/oliver-sommer/pytorch-template
[copier]: https://copier.readthedocs.io/

## Credits

- **R2-Dreamer paper** — Naoki Morihira, Amal Nahar, Kartik Bharadwaj, Yasuhiro
  Kato, Akinobu Hayashi, Tatsuya Harada. *R2-Dreamer: Redundancy-Reduced World
  Models without Decoders or Augmentation.* ICLR 2026. The full PDF lives at
  [`docs/17677_R2_Dreamer_Redundancy_Re.pdf`](docs/17677_R2_Dreamer_Redundancy_Re.pdf).
- **Reference implementation** — [`NM512/r2dreamer`][r2dreamer_code] (the official
  research repository; algorithm code in this project was reimplemented from it).
- **YAM MJCF** — the [`i2rt_yam`][i2rt_yam] model from
  [`google-deepmind/mujoco_menagerie`][menagerie], courtesy of i2rt and the
  MuJoCo Menagerie maintainers.
- **DeepMind Control Suite** — Tassa et al., [`dm_control`][dmc], the MuJoCo
  benchmark wrapper used for sanity-check tasks.

[r2dreamer_paper]: https://openreview.net/forum?id=Je2QqXrcQq
[r2dreamer_code]: https://github.com/NM512/r2dreamer
[i2rt_yam]: https://github.com/google-deepmind/mujoco_menagerie/tree/main/i2rt_yam
[menagerie]: https://github.com/google-deepmind/mujoco_menagerie
[dmc]: https://github.com/google-deepmind/dm_control
