import pytest

from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
    PitcherNormalizationConfig,
)
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig
from fantasy_baseball_manager.pipeline.stages.regression_constants import (
    BATTING_REGRESSION_PA,
    PITCHING_REGRESSION_OUTS,
)


class TestRegressionConfigDefaults:
    def test_batting_defaults_match_constants(self) -> None:
        config = RegressionConfig()
        assert config.batting_regression_pa == BATTING_REGRESSION_PA

    def test_pitching_defaults_match_constants(self) -> None:
        config = RegressionConfig()
        assert config.pitching_regression_outs == PITCHING_REGRESSION_OUTS

    def test_pitcher_normalization_defaults(self) -> None:
        config = RegressionConfig()
        assert config.pitcher_normalization == PitcherNormalizationConfig()


class TestRegressionConfigOverrides:
    def test_custom_batting_regression(self) -> None:
        custom = {"hr": 999.0}
        config = RegressionConfig(batting_regression_pa=custom)
        assert config.batting_regression_pa == {"hr": 999.0}
        assert config.pitching_regression_outs == PITCHING_REGRESSION_OUTS

    def test_custom_pitching_regression(self) -> None:
        custom = {"so": 50.0}
        config = RegressionConfig(pitching_regression_outs=custom)
        assert config.pitching_regression_outs == {"so": 50.0}
        assert config.batting_regression_pa == BATTING_REGRESSION_PA

    def test_custom_pitcher_normalization(self) -> None:
        norm = PitcherNormalizationConfig(babip_regression_weight=0.8)
        config = RegressionConfig(pitcher_normalization=norm)
        assert config.pitcher_normalization.babip_regression_weight == 0.8


class TestRegressionConfigFrozen:
    def test_cannot_reassign_attribute(self) -> None:
        config = RegressionConfig()
        with pytest.raises(AttributeError):
            config.batting_regression_pa = {}  # type: ignore[misc]


class TestRegressionConfigIndependence:
    def test_instances_have_independent_dicts(self) -> None:
        a = RegressionConfig()
        b = RegressionConfig()
        a.batting_regression_pa["hr"] = 9999.0
        assert b.batting_regression_pa["hr"] == BATTING_REGRESSION_PA["hr"]

    def test_default_not_shared_with_constant(self) -> None:
        config = RegressionConfig()
        config.batting_regression_pa["hr"] = 9999.0
        assert BATTING_REGRESSION_PA["hr"] != 9999.0
