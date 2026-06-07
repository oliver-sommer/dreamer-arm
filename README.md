# dreamer-arm

A Dreamer (**R2-Dreamer** / **DreamerV3**) implementation for the
[**i2rt YAM**][i2rt_yam] bimanual robotic arm, trained in MuJoCo via the
[DeepMind Control Suite][dmc] stack.

The world model is configurable behind a single config switch:

- `model=r2dreamer` *(default)* — decoder-free, augmentation-free. Trains the
  RSSM with a Barlow-Twins redundancy-reduction loss between the image encoder
  embedding and a projector over the latent state (per [Morihira et al., 2026][r2dreamer_paper]).
- `model=dreamerv3` — classic decoder-based reconstruction baseline.

Everything else (RSSM with categorical stochastic state, KL balancing,
actor–critic with λ-returns, LaProp + adaptive gradient clipping) is shared.

## Quick start

```bash
# install (init submodule first; TODO opt-A: drop this line once metaworld is a git URL dep)
git submodule update --init --recursive
pixi install

# train R2-Dreamer on the YAM reach task (default)
pixi run train

# train DreamerV3 baseline on the YAM reach task
pixi run train model=dreamerv3

# sanity-check on DM Control Suite cartpole-swingup
pixi run train-dmc env.task=cartpole_swingup
```

All runs log to Weights & Biases. Set `WANDB_API_KEY` (and optionally
`WANDB_ENTITY` / `WANDB_PROJECT`) before training; if unset, the logger
automatically falls back to `mode=disabled`.

## Project layout

```
src/dreamer_arm/      package source (agent, optim, envs, data, train, utils)
configs/              Hydra configs (env/, model/)
assets/i2rt_yam/      vendored YAM MJCF + meshes
tests/                pytest suite (no GPU required)
scripts/train.py      Hydra @main entry point
```

## Development

```bash
pixi run -e dev pre-commit install  # one-time setup

pixi run -e dev lint
pixi run -e dev fmt
pixi run -e dev typecheck
pixi run -e dev test
```

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
