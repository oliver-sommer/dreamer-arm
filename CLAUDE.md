# dreamer-arm

Dreamer (R2-Dreamer / DreamerV3) for the i2rt YAM arm in MuJoCo, via Meta-World.

## Layout

`configs/` and `tests/` mirror `src/dreamer_arm/`, and so does the composed
config: `configs/core/model/` lands at `cfg.core.model`, `configs/envs/arm/` at
`cfg.envs.arm`.

- `core/` — `model.py` (the `Dreamer` agent), `world_model/` (RSSM / DINO-WM,
  each behind the `WorldModel` protocol in `world_model/protocol.py`),
  `actor_critic.py`, `frozen.py` (no-grad rollout views), networks,
  distributions, losses, replay buffer, `optim/` (LaProp, AGC, `OptimStep`)
- `envs/` — Meta-World wrapper, arm plugins (`arms/`), IK, vector env
- `training/` — `trainer.py` (the loop) + `dreamer.py` (composition root)
- `inference/` — `evaluate.py`, shared by the in-loop eval and the standalone one
- `utils/` — config dispatch, console logging, W&B tracking, seeding

## Entrypoints

Each command is one config naming its own `entrypoint._target_`, which
`utils/config.py::dispatch` calls after `validate_config`. Adding a command
means adding a config plus the module it points at.

```
pixi run training     # python -m dreamer_arm.training
pixi run inference    # python -m dreamer_arm.inference.evaluate
pixi run -e dev check # format-check + lint + typecheck + test
```

`pixi run -e dev test` must keep `KMP_DUPLICATE_LIB_OK=TRUE`: torch and MuJoCo
each ship an OpenMP runtime and the loader aborts without it.

## Template

Generated from `pytorch-template` and tracked with Copier; answers live in
`configs/.copier-template.yaml`. Pull template changes with
`pixi run -e dev template-update` (needs a clean worktree).

## Notes

- The YAM MJCF and meshes live in the `thirdparty/metaworld` submodule, not in
  this repo.
- `scripts/` holds two pre-rewrite diagnostic scripts that target the old
  `EEController` / `MetaWorld` API and do not run against the current code.
