import pytest

from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.presets import (
    PIPELINES,
    marcel_full_pipeline,
    marcel_norm_pipeline,
    marcel_park_pipeline,
    marcel_pipeline,
    marcel_plus_pipeline,
    marcel_statreg_pipeline,
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
    @pytest.mark.parametrize("name", ["marcel", "marcel_park", "marcel_statreg", "marcel_plus", "marcel_norm", "marcel_full"])
    def test_registry_contains_preset(self, name: str) -> None:
        assert name in PIPELINES

    @pytest.mark.parametrize("name", ["marcel", "marcel_park", "marcel_statreg", "marcel_plus", "marcel_norm", "marcel_full"])
    def test_registry_factory_creates_pipeline(self, name: str) -> None:
        pipeline = PIPELINES[name]()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == name
