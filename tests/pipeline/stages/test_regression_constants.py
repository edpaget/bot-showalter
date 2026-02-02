from fantasy_baseball_manager.marcel.league_averages import (
    BATTING_COMPONENT_STATS,
    PITCHING_COMPONENT_STATS,
)
from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
    PITCHING_REGRESSION_OUTS,
)


class TestBattingRegressionConstants:
    def test_all_batting_stats_have_entries(self) -> None:
        for stat in BATTING_COMPONENT_STATS:
            assert stat in BATTING_REGRESSION_PA, f"Missing regression constant for batting stat: {stat}"

    def test_all_values_positive(self) -> None:
        for stat, value in BATTING_REGRESSION_PA.items():
            assert value > 0, f"Regression PA for {stat} must be positive, got {value}"


class TestPitchingRegressionConstants:
    def test_all_pitching_stats_have_entries(self) -> None:
        for stat in PITCHING_COMPONENT_STATS:
            assert stat in PITCHING_REGRESSION_OUTS, f"Missing regression constant for pitching stat: {stat}"

    def test_all_values_positive(self) -> None:
        for stat, value in PITCHING_REGRESSION_OUTS.items():
            assert value > 0, f"Regression outs for {stat} must be positive, got {value}"
