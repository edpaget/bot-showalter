from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.presets import PIPELINES, marcel_pipeline


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
