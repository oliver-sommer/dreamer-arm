"""``python -m dreamer_arm.training`` (also ``pixi run train``)."""

from dreamer_arm.utils.config import dispatch, run_hydra


def run() -> object:
    return run_hydra(dispatch, config_name="training/dreamer", selector=("training", "training"))


if __name__ == "__main__":
    run()
