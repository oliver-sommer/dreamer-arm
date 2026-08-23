import pytest

from dreamer_arm.envs.arms import ArmConfig, make_arm


def test_make_arm_selects_implementation() -> None:
    assert make_arm("yam").name == "yam"
    assert make_arm("sawyer").name == "sawyer"


def test_make_arm_preserves_config() -> None:
    arm = make_arm("yam", ArmConfig(name="yam", damping=0.25))
    assert arm._cfg.damping == pytest.approx(0.25)


def test_make_arm_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown arm"):
        make_arm("panda")


def test_sawyer_exposes_no_controller_state() -> None:
    arm = make_arm("sawyer")
    assert arm.last_diagnostics is None
    assert arm.servo_state is None
