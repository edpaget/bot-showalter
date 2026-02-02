import pytest

from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.presets import (
    PIPELINES,
    build_pipeline,
    marcel_full_pipeline,
    marcel_norm_pipeline,
    marcel_park_pipeline,
    marcel_pipeline,
    marcel_plus_pipeline,
    marcel_statreg_pipeline,
)
from fantasy_baseball_manager.pipeline.stages.component_aging import (
    ComponentAgingAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
    PitcherNormalizationAdjuster,
    PitcherNormalizationConfig,
)
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig
from fantasy_baseball_manager.pipeline.stages.stat_specific_rate_computer import (
    StatSpecificRegressionRateComputer,
)


class TestPresets:
    def test_marcel_pipeline_returns_pipeline(self) -> None:
        pipeline = marcel_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel"
        assert pipeline.years_back == 3

    def test_pipelines_registry_contains_marcel(self) -> None:
        assert "marcel" in PIPELINES

    def test_registry_factory_returns_pipeline(self) -> None:
        factory = PIPELINES["marcel"]
        pipeline = factory()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel"


class TestMarcelParkPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_park_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_park"
        assert pipeline.years_back == 3

    def test_has_three_adjusters(self) -> None:
        pipeline = marcel_park_pipeline()
        assert len(pipeline.adjusters) == 3

    def test_in_registry(self) -> None:
        assert "marcel_park" in PIPELINES


class TestMarcelStatregPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_statreg_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_statreg"
        assert pipeline.years_back == 3

    def test_has_two_adjusters(self) -> None:
        pipeline = marcel_statreg_pipeline()
        assert len(pipeline.adjusters) == 2

    def test_in_registry(self) -> None:
        assert "marcel_statreg" in PIPELINES


class TestMarcelPlusPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_plus_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_plus"
        assert pipeline.years_back == 3

    def test_has_three_adjusters(self) -> None:
        pipeline = marcel_plus_pipeline()
        assert len(pipeline.adjusters) == 3

    def test_in_registry(self) -> None:
        assert "marcel_plus" in PIPELINES


class TestMarcelNormPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_norm_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_norm"
        assert pipeline.years_back == 3

    def test_has_three_adjusters(self) -> None:
        pipeline = marcel_norm_pipeline()
        assert len(pipeline.adjusters) == 3

    def test_in_registry(self) -> None:
        assert "marcel_norm" in PIPELINES


class TestMarcelFullPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_full_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_full"
        assert pipeline.years_back == 3

    def test_has_four_adjusters(self) -> None:
        pipeline = marcel_full_pipeline()
        assert len(pipeline.adjusters) == 4

    def test_in_registry(self) -> None:
        assert "marcel_full" in PIPELINES


class TestAllPresetsInRegistry:
    @pytest.mark.parametrize(
        "name", ["marcel", "marcel_park", "marcel_statreg", "marcel_plus", "marcel_norm", "marcel_full"]
    )
    def test_registry_contains_preset(self, name: str) -> None:
        assert name in PIPELINES

    @pytest.mark.parametrize(
        "name", ["marcel", "marcel_park", "marcel_statreg", "marcel_plus", "marcel_norm", "marcel_full"]
    )
    def test_registry_factory_creates_pipeline(self, name: str) -> None:
        pipeline = PIPELINES[name]()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == name

    @pytest.mark.parametrize(
        "name", ["marcel", "marcel_park", "marcel_statreg", "marcel_plus", "marcel_norm", "marcel_full"]
    )
    def test_aging_adjuster_is_component_aging(self, name: str) -> None:
        pipeline = PIPELINES[name]()
        aging_adjusters = [a for a in pipeline.adjusters if isinstance(a, ComponentAgingAdjuster)]
        assert len(aging_adjusters) == 1


class TestConfigThreading:
    """Verify that RegressionConfig threads through to pipeline components."""

    def test_zero_arg_calls_produce_default_pipelines(self) -> None:
        for factory in [marcel_statreg_pipeline, marcel_plus_pipeline, marcel_norm_pipeline, marcel_full_pipeline]:
            pipeline = factory()
            assert isinstance(pipeline, ProjectionPipeline)

    @pytest.mark.parametrize(
        "factory",
        [marcel_statreg_pipeline, marcel_plus_pipeline, marcel_norm_pipeline, marcel_full_pipeline],
    )
    def test_custom_config_threads_to_rate_computer(
        self,
        factory: object,
    ) -> None:
        custom_batting = {"hr": 999.0}
        config = RegressionConfig(batting_regression_pa=custom_batting)
        pipeline = factory(config=config)  # type: ignore[operator]
        assert isinstance(pipeline.rate_computer, StatSpecificRegressionRateComputer)
        assert pipeline.rate_computer._batting_regression == custom_batting

    @pytest.mark.parametrize("factory", [marcel_norm_pipeline, marcel_full_pipeline])
    def test_custom_config_threads_to_normalization_adjuster(
        self,
        factory: object,
    ) -> None:
        norm = PitcherNormalizationConfig(babip_regression_weight=0.99)
        config = RegressionConfig(pitcher_normalization=norm)
        pipeline = factory(config=config)  # type: ignore[operator]
        norm_adjusters = [a for a in pipeline.adjusters if isinstance(a, PitcherNormalizationAdjuster)]
        assert len(norm_adjusters) == 1
        assert norm_adjusters[0]._config.babip_regression_weight == 0.99


class TestBuildPipeline:
    @pytest.mark.parametrize("name", list(PIPELINES.keys()))
    def test_dispatches_all_registered_pipelines(self, name: str) -> None:
        pipeline = build_pipeline(name)
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == name

    def test_passes_config_to_configurable_pipeline(self) -> None:
        custom = {"hr": 123.0}
        config = RegressionConfig(batting_regression_pa=custom)
        pipeline = build_pipeline("marcel_norm", config=config)
        assert isinstance(pipeline.rate_computer, StatSpecificRegressionRateComputer)
        assert pipeline.rate_computer._batting_regression == custom

    def test_ignores_config_for_non_configurable_pipeline(self) -> None:
        config = RegressionConfig(batting_regression_pa={"hr": 123.0})
        pipeline = build_pipeline("marcel", config=config)
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel"

    def test_raises_for_unknown_pipeline(self) -> None:
        with pytest.raises(ValueError, match="Unknown pipeline"):
            build_pipeline("nonexistent")
