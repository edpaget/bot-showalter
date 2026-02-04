import pytest

from fantasy_baseball_manager.pipeline.builder import PipelineBuilder
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.stages.adjusters import RebaselineAdjuster
from fantasy_baseball_manager.pipeline.stages.component_aging import (
    ComponentAgingAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig
from fantasy_baseball_manager.pipeline.stages.stat_specific_rate_computer import (
    StatSpecificRegressionRateComputer,
)
from fantasy_baseball_manager.pipeline.statcast_data import StatcastBatterStats


class FakeStatcastSource:
    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        return []

    def pitcher_expected_stats(self, year: int) -> list:
        return []


class FakeIdMapper:
    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return None

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return None

    def fangraphs_to_mlbam(self, fangraphs_id: str) -> str | None:
        return None

    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
        return None


class TestPipelineBuilderDefaults:
    def test_default_build_produces_pipeline(self) -> None:
        pipeline = PipelineBuilder().build()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "custom"

    def test_default_has_two_adjusters(self) -> None:
        pipeline = PipelineBuilder().build()
        assert len(pipeline.adjusters) == 2
        assert isinstance(pipeline.adjusters[0], RebaselineAdjuster)
        assert isinstance(pipeline.adjusters[1], ComponentAgingAdjuster)

    def test_default_uses_stat_specific_rate_computer(self) -> None:
        pipeline = PipelineBuilder().build()
        assert isinstance(pipeline.rate_computer, StatSpecificRegressionRateComputer)


class TestPipelineBuilderWithParkFactors:
    def test_adds_park_factor_adjuster(self) -> None:
        pipeline = PipelineBuilder().with_park_factors().build()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "ParkFactorAdjuster" in adjuster_types

    def test_park_factor_before_rebaseline(self) -> None:
        pipeline = PipelineBuilder().with_park_factors().build()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        pf_idx = adjuster_types.index("ParkFactorAdjuster")
        rb_idx = adjuster_types.index("RebaselineAdjuster")
        assert pf_idx < rb_idx


class TestPipelineBuilderWithPitcherNorm:
    def test_adds_pitcher_normalization(self) -> None:
        pipeline = PipelineBuilder().with_pitcher_normalization().build()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "PitcherNormalizationAdjuster" in adjuster_types


class TestPipelineBuilderWithStatcast:
    def test_adds_statcast_adjuster(self) -> None:
        pipeline = (
            PipelineBuilder()
            .with_statcast(statcast_source=FakeStatcastSource(), id_mapper=FakeIdMapper())
            .build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "StatcastRateAdjuster" in adjuster_types

    def test_statcast_before_rebaseline(self) -> None:
        pipeline = (
            PipelineBuilder()
            .with_statcast(statcast_source=FakeStatcastSource(), id_mapper=FakeIdMapper())
            .build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        sc_idx = adjuster_types.index("StatcastRateAdjuster")
        rb_idx = adjuster_types.index("RebaselineAdjuster")
        assert sc_idx < rb_idx


class TestPipelineBuilderWithBatterBabip:
    def test_adds_batter_babip_adjuster(self) -> None:
        pipeline = (
            PipelineBuilder()
            .with_batter_babip(statcast_source=FakeStatcastSource(), id_mapper=FakeIdMapper())
            .build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "BatterBabipAdjuster" in adjuster_types


class TestPipelineBuilderWithPitcherStatcast:
    def test_adds_pitcher_statcast_adjuster(self) -> None:
        pipeline = (
            PipelineBuilder()
            .with_pitcher_statcast(pitcher_statcast_source=FakeStatcastSource(), id_mapper=FakeIdMapper())
            .build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "PitcherStatcastAdjuster" in adjuster_types


class TestPipelineBuilderOrdering:
    def test_full_ordering(self) -> None:
        fake_source = FakeStatcastSource()
        fake_mapper = FakeIdMapper()
        pipeline = (
            PipelineBuilder()
            .with_park_factors()
            .with_pitcher_normalization()
            .with_pitcher_statcast(pitcher_statcast_source=fake_source, id_mapper=fake_mapper)
            .with_statcast(statcast_source=fake_source, id_mapper=fake_mapper)
            .with_batter_babip(statcast_source=fake_source, id_mapper=fake_mapper)
            .build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        expected_order = [
            "ParkFactorAdjuster",
            "PitcherNormalizationAdjuster",
            "PitcherStatcastAdjuster",
            "StatcastRateAdjuster",
            "BatterBabipAdjuster",
            "RebaselineAdjuster",
            "ComponentAgingAdjuster",
        ]
        assert adjuster_types == expected_order


class TestPipelineBuilderCustomConfig:
    def test_config_threads_to_rate_computer(self) -> None:
        custom_batting = {"hr": 999.0}
        config = RegressionConfig(batting_regression_pa=custom_batting)
        pipeline = PipelineBuilder(config=config).build()
        assert isinstance(pipeline.rate_computer, StatSpecificRegressionRateComputer)
        assert pipeline.rate_computer._batting_regression == custom_batting

    def test_custom_name(self) -> None:
        pipeline = PipelineBuilder("my_pipeline").build()
        assert pipeline.name == "my_pipeline"


class TestPipelineBuilderRateComputer:
    def test_unknown_rate_computer_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown rate computer"):
            PipelineBuilder().rate_computer("unknown")

    def test_platoon_rate_computer(self) -> None:
        from fantasy_baseball_manager.marcel.models import BattingSeasonStats
        from fantasy_baseball_manager.pipeline.stages.platoon_rate_computer import (
            PlatoonRateComputer,
        )

        class FakeSplitSource:
            def batting_stats_vs_lhp(self, year: int) -> list[BattingSeasonStats]:
                return []

            def batting_stats_vs_rhp(self, year: int) -> list[BattingSeasonStats]:
                return []

        pipeline = (
            PipelineBuilder()
            .rate_computer("platoon")
            .with_split_source(FakeSplitSource())
            .build()
        )
        assert isinstance(pipeline.rate_computer, PlatoonRateComputer)


class FakeCacheStore:
    """A fake cache store for testing dependency injection."""

    def __init__(self) -> None:
        self.get_calls: list[tuple[str, str]] = []
        self.put_calls: list[tuple[str, str]] = []

    def get(self, namespace: str, key: str) -> str | None:
        self.get_calls.append((namespace, key))
        return None

    def put(self, namespace: str, key: str, value: str, ttl_seconds: int) -> None:
        self.put_calls.append((namespace, key))

    def invalidate(self, namespace: str, key: str | None = None) -> None:
        pass


class TestPipelineBuilderCacheStore:
    def test_cache_store_via_constructor(self) -> None:
        """Test that cache_store can be injected via constructor."""
        fake_cache = FakeCacheStore()
        pipeline = PipelineBuilder(cache_store=fake_cache).with_park_factors().build()
        assert "ParkFactorAdjuster" in [type(a).__name__ for a in pipeline.adjusters]

    def test_cache_store_via_builder_method(self) -> None:
        """Test that cache_store can be injected via with_cache_store()."""
        fake_cache = FakeCacheStore()
        pipeline = (
            PipelineBuilder()
            .with_cache_store(fake_cache)
            .with_park_factors()
            .build()
        )
        assert "ParkFactorAdjuster" in [type(a).__name__ for a in pipeline.adjusters]
