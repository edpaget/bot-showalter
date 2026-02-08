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
from fantasy_baseball_manager.player_id.mapper import SfbbMapper


class FakeStatcastSource:
    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        return []

    def pitcher_expected_stats(self, year: int) -> list:
        return []


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


def _fake_mapper() -> SfbbMapper:
    return SfbbMapper({}, {})


class TestPipelineBuilderWithStatcast:
    def test_adds_statcast_adjuster(self) -> None:
        pipeline = PipelineBuilder(id_mapper=_fake_mapper()).with_statcast(statcast_source=FakeStatcastSource()).build()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "StatcastRateAdjuster" in adjuster_types

    def test_statcast_before_rebaseline(self) -> None:
        pipeline = PipelineBuilder(id_mapper=_fake_mapper()).with_statcast(statcast_source=FakeStatcastSource()).build()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        sc_idx = adjuster_types.index("StatcastRateAdjuster")
        rb_idx = adjuster_types.index("RebaselineAdjuster")
        assert sc_idx < rb_idx


class TestPipelineBuilderWithBatterBabip:
    def test_adds_batter_babip_adjuster(self) -> None:
        pipeline = (
            PipelineBuilder(id_mapper=_fake_mapper()).with_batter_babip(statcast_source=FakeStatcastSource()).build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "BatterBabipAdjuster" in adjuster_types


class TestPipelineBuilderWithPitcherStatcast:
    def test_adds_pitcher_statcast_adjuster(self) -> None:
        pipeline = (
            PipelineBuilder(id_mapper=_fake_mapper())
            .with_pitcher_statcast(pitcher_statcast_source=FakeStatcastSource())
            .build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "PitcherStatcastAdjuster" in adjuster_types


class TestPipelineBuilderOrdering:
    def test_full_ordering(self) -> None:
        fake_source = FakeStatcastSource()
        pipeline = (
            PipelineBuilder(id_mapper=_fake_mapper())
            .with_park_factors()
            .with_pitcher_normalization()
            .with_pitcher_statcast(pitcher_statcast_source=fake_source)
            .with_statcast(statcast_source=fake_source)
            .with_batter_babip(statcast_source=fake_source)
            .build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        expected_order = [
            "PlayerIdentityEnricher",
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

        pipeline = PipelineBuilder().rate_computer("platoon").with_split_source(FakeSplitSource()).build()
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
        pipeline = PipelineBuilder().with_cache_store(fake_cache).with_park_factors().build()
        assert "ParkFactorAdjuster" in [type(a).__name__ for a in pipeline.adjusters]


class TestPipelineBuilderWithContextual:
    def test_with_contextual_uses_contextual_rate_computer(self) -> None:
        from fantasy_baseball_manager.pipeline.stages.contextual_rate_computer import (
            ContextualEmbeddingRateComputer,
        )

        pipeline = PipelineBuilder(id_mapper=_fake_mapper()).with_contextual().build()
        assert isinstance(pipeline.rate_computer, ContextualEmbeddingRateComputer)

    def test_with_contextual_custom_config(self) -> None:
        from fantasy_baseball_manager.contextual.training.config import (
            ContextualRateComputerConfig,
        )
        from fantasy_baseball_manager.pipeline.stages.contextual_rate_computer import (
            ContextualEmbeddingRateComputer,
        )

        config = ContextualRateComputerConfig(min_games=20, context_window=15)
        pipeline = PipelineBuilder(id_mapper=_fake_mapper()).with_contextual(config=config).build()
        assert isinstance(pipeline.rate_computer, ContextualEmbeddingRateComputer)
        assert pipeline.rate_computer.config.min_games == 20
        assert pipeline.rate_computer.config.context_window == 15

    def test_with_contextual_adds_identity_enricher(self) -> None:
        pipeline = PipelineBuilder(id_mapper=_fake_mapper()).with_contextual().build()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "PlayerIdentityEnricher" in adjuster_types

    def test_with_contextual_and_park_factors(self) -> None:
        pipeline = (
            PipelineBuilder(id_mapper=_fake_mapper())
            .with_contextual()
            .with_park_factors()
            .with_pitcher_normalization()
            .build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "PlayerIdentityEnricher" in adjuster_types
        assert "ParkFactorAdjuster" in adjuster_types
        assert "PitcherNormalizationAdjuster" in adjuster_types
        assert "RebaselineAdjuster" in adjuster_types
        assert "ComponentAgingAdjuster" in adjuster_types


class TestPipelineBuilderWithContextualBlender:
    def test_with_contextual_blender_adds_blender(self) -> None:
        pipeline = PipelineBuilder(id_mapper=_fake_mapper()).with_contextual_blender().build()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "ContextualBlender" in adjuster_types

    def test_blender_before_rebaseline(self) -> None:
        pipeline = PipelineBuilder(id_mapper=_fake_mapper()).with_contextual_blender().build()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        cb_idx = adjuster_types.index("ContextualBlender")
        rb_idx = adjuster_types.index("RebaselineAdjuster")
        assert cb_idx < rb_idx

    def test_identity_enricher_included(self) -> None:
        pipeline = PipelineBuilder(id_mapper=_fake_mapper()).with_contextual_blender().build()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "PlayerIdentityEnricher" in adjuster_types

    def test_custom_config_threads_through(self) -> None:
        from fantasy_baseball_manager.contextual.training.config import (
            ContextualBlenderConfig,
        )
        from fantasy_baseball_manager.pipeline.stages.contextual_blender import (
            ContextualBlender,
        )

        config = ContextualBlenderConfig(contextual_weight=0.5, min_games=20)
        pipeline = PipelineBuilder(id_mapper=_fake_mapper()).with_contextual_blender(config=config).build()
        blenders = [a for a in pipeline.adjusters if isinstance(a, ContextualBlender)]
        assert len(blenders) == 1
        assert blenders[0].config.contextual_weight == 0.5
        assert blenders[0].config.min_games == 20

    def test_with_all_marcel_full_adjusters(self) -> None:
        """Contextual blender with all marcel_full adjusters."""
        fake_source = FakeStatcastSource()
        pipeline = (
            PipelineBuilder(id_mapper=_fake_mapper())
            .with_park_factors()
            .with_pitcher_normalization()
            .with_pitcher_statcast(pitcher_statcast_source=fake_source)
            .with_statcast(statcast_source=fake_source)
            .with_batter_babip(statcast_source=fake_source)
            .with_contextual_blender()
            .build()
        )
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "ContextualBlender" in adjuster_types
        assert "BatterBabipAdjuster" in adjuster_types
        # Blender should be after batter_babip but before rebaseline
        cb_idx = adjuster_types.index("ContextualBlender")
        bb_idx = adjuster_types.index("BatterBabipAdjuster")
        rb_idx = adjuster_types.index("RebaselineAdjuster")
        assert bb_idx < cb_idx < rb_idx
