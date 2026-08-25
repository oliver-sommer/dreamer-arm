from dreamer_arm.envs.sim.arms import ArmConfig
from dreamer_arm.envs.sim.controller_bench import DiagnosticResult, _build_env, _smooth_push_episode


def test_diagnostic_result_has_uniform_metric_shape() -> None:
    result = DiagnosticResult(mode="coverage", metrics={"reach/covered": 2.0, "reach/total": 3.0})
    assert result.mode == "coverage"
    assert all(isinstance(value, float) for value in result.metrics.values())


def test_smooth_yam_push_reaches_success_without_policy_chatter() -> None:
    # The generic controller default is deliberately position-only while the
    # current reach failure is isolated.  This staged contact diagnostic needs
    # a modest orientation constraint to keep the fingertips behind the puck
    # and a longer following-error leash for sustained horizontal contact.
    # The generic 0.035 m default remains covered by the reach/reversal bench;
    # this push-specific probe explicitly requests its measured 0.06 m need.
    inner, arm, gid = _build_env(
        ArmConfig(name="yam", damping=0.15, ori_weight=0.3, max_lag_m=0.06),
        "yam",
        "push",
        seed=0,
    )
    try:
        metrics = _smooth_push_episode(inner, arm, gid)
    finally:
        inner.close()

    assert metrics["success"] == 1.0
    assert metrics["puck_motion_m"] > 0.1
