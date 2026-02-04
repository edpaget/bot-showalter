from fantasy_baseball_manager.marcel.league_averages import BATTING_COMPONENT_STATS
from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
)
from fantasy_baseball_manager.pipeline.stages.split_regression_constants import (
    BATTING_SPLIT_REGRESSION_PA,
)


class TestBattingSplitRegressionConstants:
    def test_all_batting_stats_have_entries(self) -> None:
        for stat in BATTING_COMPONENT_STATS:
            assert stat in BATTING_SPLIT_REGRESSION_PA, f"Missing split regression constant for batting stat: {stat}"

    def test_all_values_positive(self) -> None:
        for stat, value in BATTING_SPLIT_REGRESSION_PA.items():
            assert value > 0, f"Split regression PA for {stat} must be positive, got {value}"

    def test_each_value_is_double_base(self) -> None:
        for stat, value in BATTING_SPLIT_REGRESSION_PA.items():
            expected = BATTING_REGRESSION_PA[stat] * 2
            assert value == expected, f"Split regression PA for {stat} should be {expected}, got {value}"

    def test_same_keys_as_base(self) -> None:
        assert set(BATTING_SPLIT_REGRESSION_PA.keys()) == set(BATTING_REGRESSION_PA.keys())
