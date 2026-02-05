import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.minors.rate_computer import MLERateComputer
from fantasy_baseball_manager.pipeline.presets import (
    PIPELINES,
    build_pipeline,
    marcel_classic_pipeline,
    marcel_full_pipeline,
    marcel_gb_pipeline,
    marcel_pipeline,
    mle_pipeline,
)
from fantasy_baseball_manager.pipeline.stages.component_aging import (
    ComponentAgingAdjuster,
)
from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
    PitcherNormalizationConfig,
)
from fantasy_baseball_manager.pipeline.stages.regression_config import RegressionConfig
from fantasy_baseball_manager.pipeline.stages.stat_specific_rate_computer import (
    StatSpecificRegressionRateComputer,
)


class TestMarcelClassicPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_classic_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_classic"
        assert pipeline.years_back == 3

    def test_has_two_adjusters(self) -> None:
        pipeline = marcel_classic_pipeline()
        assert len(pipeline.adjusters) == 2

    def test_in_registry(self) -> None:
        assert "marcel_classic" in PIPELINES


class TestMarcelPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel"
        assert pipeline.years_back == 3

    def test_has_two_adjusters(self) -> None:
        pipeline = marcel_pipeline()
        assert len(pipeline.adjusters) == 2

    def test_uses_stat_specific_rate_computer(self) -> None:
        pipeline = marcel_pipeline()
        assert isinstance(pipeline.rate_computer, StatSpecificRegressionRateComputer)

    def test_in_registry(self) -> None:
        assert "marcel" in PIPELINES


class TestMarcelFullPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_full_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_full"
        assert pipeline.years_back == 3

    def test_has_seven_adjusters(self) -> None:
        # park, pitcher_norm, pitcher_statcast, statcast, batter_babip, rebaseline, aging
        pipeline = marcel_full_pipeline()
        assert len(pipeline.adjusters) == 7

    def test_ordering_park_pitcher_statcast_rebaseline(self) -> None:
        pipeline = marcel_full_pipeline()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        pf_idx = adjuster_types.index("ParkFactorAdjuster")
        pn_idx = adjuster_types.index("PitcherNormalizationAdjuster")
        ps_idx = adjuster_types.index("PitcherStatcastAdjuster")
        sc_idx = adjuster_types.index("StatcastRateAdjuster")
        bb_idx = adjuster_types.index("BatterBabipAdjuster")
        rb_idx = adjuster_types.index("RebaselineAdjuster")
        ca_idx = adjuster_types.index("ComponentAgingAdjuster")
        assert pf_idx < pn_idx < ps_idx < sc_idx < bb_idx < rb_idx < ca_idx

    def test_in_registry(self) -> None:
        assert "marcel_full" in PIPELINES


class TestMarcelGBPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_gb_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_gb"
        assert pipeline.years_back == 3

    def test_has_eight_adjusters(self) -> None:
        # park, pitcher_norm, pitcher_statcast, statcast, batter_babip, gb_residual, rebaseline, aging
        pipeline = marcel_gb_pipeline()
        assert len(pipeline.adjusters) == 8

    def test_includes_gb_residual_adjuster(self) -> None:
        pipeline = marcel_gb_pipeline()
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        assert "GBResidualAdjuster" in adjuster_types

    def test_in_registry(self) -> None:
        assert "marcel_gb" in PIPELINES


class TestMLEPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = mle_pipeline()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "mle"
        assert pipeline.years_back == 3

    def test_has_two_adjusters(self) -> None:
        """MLE should have RebaselineAdjuster and ComponentAgingAdjuster."""
        pipeline = mle_pipeline()
        assert len(pipeline.adjusters) == 2

    def test_uses_mle_rate_computer(self) -> None:
        """MLE should use MLERateComputer."""
        pipeline = mle_pipeline()
        assert isinstance(pipeline.rate_computer, MLERateComputer)

    def test_in_registry(self) -> None:
        """MLE should be in PIPELINES registry."""
        assert "mle" in PIPELINES


ALL_PRESET_NAMES = [
    "marcel_classic",
    "marcel",
    "marcel_full",
    "marcel_gb",
    "marcel_gb_mle",
    "mtl",
    "marcel_mtl",
    "mle",
]


class TestAllPresetsInRegistry:
    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_registry_contains_preset(self, name: str) -> None:
        assert name in PIPELINES

    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_registry_factory_creates_pipeline(self, name: str) -> None:
        pipeline = PIPELINES[name]()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == name

    @pytest.mark.parametrize("name", ["marcel", "marcel_full", "marcel_gb"])
    def test_aging_adjuster_is_component_aging(self, name: str) -> None:
        pipeline = PIPELINES[name]()
        aging_adjusters = [a for a in pipeline.adjusters if isinstance(a, ComponentAgingAdjuster)]
        assert len(aging_adjusters) == 1

    def test_registry_has_expected_entries(self) -> None:
        assert len(PIPELINES) == 10  # 8 internal + steamer + zips


class TestConfigThreading:
    """Verify that RegressionConfig threads through to pipeline components."""

    def test_zero_arg_calls_produce_default_pipelines(self) -> None:
        for factory in [marcel_pipeline, marcel_full_pipeline, marcel_gb_pipeline]:
            pipeline = factory()
            assert isinstance(pipeline, ProjectionPipeline)

    @pytest.mark.parametrize("factory", [marcel_pipeline, marcel_full_pipeline, marcel_gb_pipeline])
    def test_custom_config_threads_to_rate_computer(
        self,
        factory: object,
    ) -> None:
        custom_batting = {"hr": 999.0}
        config = RegressionConfig(batting_regression_pa=custom_batting)
        pipeline = factory(config=config)  # type: ignore[operator]
        assert isinstance(pipeline.rate_computer, StatSpecificRegressionRateComputer)
        assert pipeline.rate_computer._batting_regression == custom_batting

    def test_custom_config_threads_to_normalization_adjuster_in_full(self) -> None:
        from fantasy_baseball_manager.pipeline.stages.pitcher_normalization import (
            PitcherNormalizationAdjuster,
        )

        norm = PitcherNormalizationConfig(babip_regression_weight=0.99)
        config = RegressionConfig(pitcher_normalization=norm)
        pipeline = marcel_full_pipeline(config=config)
        norm_adjusters = [a for a in pipeline.adjusters if isinstance(a, PitcherNormalizationAdjuster)]
        assert len(norm_adjusters) == 1
        assert norm_adjusters[0]._config.babip_regression_weight == 0.99


class TestBuildPipeline:
    @pytest.mark.parametrize("name", list(PIPELINES.keys()))
    def test_dispatches_all_registered_pipelines(self, name: str) -> None:
        pipeline = build_pipeline(name)
        # Check duck typing - pipeline must have projection methods
        assert hasattr(pipeline, "project_batters")
        assert hasattr(pipeline, "project_pitchers")
        assert callable(pipeline.project_batters)
        assert callable(pipeline.project_pitchers)

    def test_passes_config_to_configurable_pipeline(self) -> None:
        custom = {"hr": 123.0}
        config = RegressionConfig(batting_regression_pa=custom)
        pipeline = build_pipeline("marcel", config=config)
        assert isinstance(pipeline.rate_computer, StatSpecificRegressionRateComputer)
        assert pipeline.rate_computer._batting_regression == custom

    def test_raises_for_unknown_pipeline(self) -> None:
        with pytest.raises(ValueError, match="Unknown pipeline"):
            build_pipeline("nonexistent")


# --- Integration test fixtures ---


def _make_batting_stats(
    player_id: str = "p1",
    name: str = "Test Hitter",
    year: int = 2024,
    age: int = 28,
    pa: int = 600,
    ab: int = 540,
    h: int = 160,
    singles: int = 100,
    doubles: int = 30,
    triples: int = 5,
    hr: int = 25,
    bb: int = 50,
    so: int = 120,
    hbp: int = 5,
    sf: int = 3,
    sh: int = 2,
    sb: int = 10,
    cs: int = 3,
    r: int = 80,
    rbi: int = 90,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=age,
        pa=pa,
        ab=ab,
        h=h,
        singles=singles,
        doubles=doubles,
        triples=triples,
        hr=hr,
        bb=bb,
        so=so,
        hbp=hbp,
        sf=sf,
        sh=sh,
        sb=sb,
        cs=cs,
        r=r,
        rbi=rbi,
    )


def _make_league_batting(year: int = 2024) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        pa=6000,
        ab=5400,
        h=1500,
        singles=900,
        doubles=300,
        triples=30,
        hr=200,
        bb=500,
        so=1400,
        hbp=50,
        sf=30,
        sh=20,
        sb=100,
        cs=30,
        r=800,
        rbi=750,
    )


class IntegrationDataSource:
    """Fake data source for full pipeline integration tests."""

    def __init__(
        self,
        player_batting: dict[int, list[BattingSeasonStats]] | None = None,
        team_batting: dict[int, list[BattingSeasonStats]] | None = None,
        player_pitching: dict[int, list[PitchingSeasonStats]] | None = None,
        team_pitching: dict[int, list[PitchingSeasonStats]] | None = None,
    ) -> None:
        self._player_batting = player_batting or {}
        self._team_batting = team_batting or {}
        self._player_pitching = player_pitching or {}
        self._team_pitching = team_pitching or {}

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._player_batting.get(year, [])

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._player_pitching.get(year, [])

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return self._team_batting.get(year, [])

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._team_pitching.get(year, [])


class TestAllPipelinesProduceValidProjections:
    """Every registered pipeline produces non-empty, non-negative batting projections."""

    def test_marcel_produces_valid_batting(self) -> None:
        league = _make_league_batting()
        ds = IntegrationDataSource(
            player_batting={
                2024: [_make_batting_stats(year=2024, age=28)],
                2023: [_make_batting_stats(year=2023, age=27)],
                2022: [_make_batting_stats(year=2022, age=26)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        pipeline = PIPELINES["marcel"]()
        result = pipeline.project_batters(ds, 2025)
        assert len(result) == 1
        proj = result[0]
        assert isinstance(proj, BattingProjection)
        assert proj.pa > 0
        assert proj.hr >= 0
        assert proj.h >= 0
        assert proj.bb >= 0

    def test_mle_produces_valid_batting(self) -> None:
        """MLE pipeline produces valid batting projections (falls back to Marcel)."""
        league = _make_league_batting()
        ds = IntegrationDataSource(
            player_batting={
                2024: [_make_batting_stats(year=2024, age=28)],
                2023: [_make_batting_stats(year=2023, age=27)],
                2022: [_make_batting_stats(year=2022, age=26)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        pipeline = PIPELINES["mle"]()
        result = pipeline.project_batters(ds, 2025)
        assert len(result) == 1
        proj = result[0]
        assert isinstance(proj, BattingProjection)
        assert proj.pa > 0
        assert proj.hr >= 0
        assert proj.h >= 0
        assert proj.bb >= 0


class TestMLEFallbackBehavior:
    """Test MLE gracefully falls back to Marcel when no model exists."""

    def test_mle_pipeline_falls_back_when_no_model(self) -> None:
        """MLE should produce valid projections even without trained model."""
        league = _make_league_batting()
        ds = IntegrationDataSource(
            player_batting={
                2024: [_make_batting_stats(year=2024, age=28)],
                2023: [_make_batting_stats(year=2023, age=27)],
                2022: [_make_batting_stats(year=2022, age=26)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        # MLE pipeline without a trained model should fall back to Marcel rates
        pipeline = mle_pipeline()
        result = pipeline.project_batters(ds, 2025)

        # Should still produce valid projections via Marcel fallback
        assert len(result) == 1
        proj = result[0]
        assert isinstance(proj, BattingProjection)
        assert proj.pa > 0
        assert proj.hr >= 0
