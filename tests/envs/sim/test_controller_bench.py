from dreamer_arm.envs.sim.controller_bench import DiagnosticResult


def test_diagnostic_result_has_uniform_metric_shape() -> None:
    result = DiagnosticResult(mode="coverage", metrics={"reach/covered": 2.0, "reach/total": 3.0})
    assert result.mode == "coverage"
    assert all(isinstance(value, float) for value in result.metrics.values())
