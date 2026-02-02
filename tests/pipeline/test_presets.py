import pytest

from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.presets import (
    PIPELINES,
    build_pipeline,
    marcel_full_pipeline,
    marcel_full_statcast_pipeline,
    marcel_norm_pipeline,
    marcel_park_pipeline,
    marcel_pipeline,
    marcel_plus_pipeline,
    marcel_plus_statcast_pipeline,
    marcel_statcast_pipeline,
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
from fantasy_baseball_manager.pipeline.stages.statcast_adjuster import (
    StatcastRateAdjuster,
)
from fantasy_baseball_manager.pipeline.statcast_data import StatcastBatterStats


class FakeStatcastSource:
    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
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


class TestMarcelStatcastPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_statcast"
        assert pipeline.years_back == 3

    def test_has_three_adjusters(self) -> None:
        pipeline = marcel_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        assert len(pipeline.adjusters) == 3

    def test_statcast_adjuster_present(self) -> None:
        pipeline = marcel_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        statcast_adjusters = [a for a in pipeline.adjusters if isinstance(a, StatcastRateAdjuster)]
        assert len(statcast_adjusters) == 1

    def test_statcast_before_rebaseline(self) -> None:
        pipeline = marcel_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        sc_idx = adjuster_types.index("StatcastRateAdjuster")
        rb_idx = adjuster_types.index("RebaselineAdjuster")
        assert sc_idx < rb_idx

    def test_in_registry(self) -> None:
        assert "marcel_statcast" in PIPELINES


class TestMarcelPlusStatcastPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_plus_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_plus_statcast"
        assert pipeline.years_back == 3

    def test_has_four_adjusters(self) -> None:
        pipeline = marcel_plus_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        assert len(pipeline.adjusters) == 4

    def test_statcast_adjuster_present(self) -> None:
        pipeline = marcel_plus_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        statcast_adjusters = [a for a in pipeline.adjusters if isinstance(a, StatcastRateAdjuster)]
        assert len(statcast_adjusters) == 1

    def test_park_factor_before_statcast(self) -> None:
        pipeline = marcel_plus_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        pf_idx = adjuster_types.index("ParkFactorAdjuster")
        sc_idx = adjuster_types.index("StatcastRateAdjuster")
        assert pf_idx < sc_idx

    def test_in_registry(self) -> None:
        assert "marcel_plus_statcast" in PIPELINES


class TestMarcelFullStatcastPreset:
    def test_returns_pipeline(self) -> None:
        pipeline = marcel_full_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == "marcel_full_statcast"
        assert pipeline.years_back == 3

    def test_has_five_adjusters(self) -> None:
        pipeline = marcel_full_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        assert len(pipeline.adjusters) == 5

    def test_statcast_adjuster_present(self) -> None:
        pipeline = marcel_full_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        statcast_adjusters = [a for a in pipeline.adjusters if isinstance(a, StatcastRateAdjuster)]
        assert len(statcast_adjusters) == 1

    def test_ordering_park_pitcher_statcast_rebaseline(self) -> None:
        pipeline = marcel_full_statcast_pipeline(FakeStatcastSource(), FakeIdMapper())
        adjuster_types = [type(a).__name__ for a in pipeline.adjusters]
        pf_idx = adjuster_types.index("ParkFactorAdjuster")
        pn_idx = adjuster_types.index("PitcherNormalizationAdjuster")
        sc_idx = adjuster_types.index("StatcastRateAdjuster")
        rb_idx = adjuster_types.index("RebaselineAdjuster")
        assert pf_idx < pn_idx < sc_idx < rb_idx

    def test_in_registry(self) -> None:
        assert "marcel_full_statcast" in PIPELINES


ALL_PRESET_NAMES = [
    "marcel",
    "marcel_park",
    "marcel_statreg",
    "marcel_plus",
    "marcel_norm",
    "marcel_full",
    "marcel_statcast",
    "marcel_plus_statcast",
    "marcel_full_statcast",
]

NON_STATCAST_PRESET_NAMES = [
    "marcel",
    "marcel_park",
    "marcel_statreg",
    "marcel_plus",
    "marcel_norm",
    "marcel_full",
]


class TestAllPresetsInRegistry:
    @pytest.mark.parametrize("name", ALL_PRESET_NAMES)
    def test_registry_contains_preset(self, name: str) -> None:
        assert name in PIPELINES

    @pytest.mark.parametrize("name", NON_STATCAST_PRESET_NAMES)
    def test_registry_factory_creates_pipeline(self, name: str) -> None:
        pipeline = PIPELINES[name]()
        assert isinstance(pipeline, ProjectionPipeline)
        assert pipeline.name == name

    @pytest.mark.parametrize("name", NON_STATCAST_PRESET_NAMES)
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


def _make_pitching_stats(
    player_id: str = "sp1",
    year: int = 2024,
    age: int = 28,
) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id=player_id,
        name="Test Pitcher",
        year=year,
        age=age,
        ip=180.0,
        g=32,
        gs=32,
        er=70,
        h=150,
        bb=50,
        so=200,
        hr=20,
        hbp=5,
        w=0,
        sv=0,
        hld=0,
        bs=0,
    )


def _make_league_pitching(year: int = 2024) -> PitchingSeasonStats:
    return PitchingSeasonStats(
        player_id="league",
        name="League Total",
        year=year,
        age=0,
        ip=1450.0,
        g=500,
        gs=162,
        er=650,
        h=1350,
        bb=500,
        so=1400,
        hr=180,
        hbp=60,
        w=0,
        sv=0,
        hld=0,
        bs=0,
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


class IntegrationStatcastSource:
    """Statcast source with realistic data for integration tests."""

    def __init__(self, data: dict[int, list[StatcastBatterStats]]) -> None:
        self._data = data

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        return self._data.get(year, [])


class IntegrationIdMapper:
    """ID mapper for integration tests."""

    def __init__(self, fg_to_mlbam: dict[str, str]) -> None:
        self._fg_to_mlbam = fg_to_mlbam
        self._mlbam_to_fg = {v: k for k, v in fg_to_mlbam.items()}

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return None

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return None

    def fangraphs_to_mlbam(self, fangraphs_id: str) -> str | None:
        return self._fg_to_mlbam.get(fangraphs_id)

    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
        return self._mlbam_to_fg.get(mlbam_id)


STATCAST_PRESET_NAMES = [
    "marcel_statcast",
    "marcel_plus_statcast",
    "marcel_full_statcast",
]

STATCAST_FACTORY_MAP = {
    "marcel_statcast": marcel_statcast_pipeline,
    "marcel_plus_statcast": marcel_plus_statcast_pipeline,
    "marcel_full_statcast": marcel_full_statcast_pipeline,
}


class TestStatcastPipelineIntegration:
    """Full pipeline integration tests with fake data sources."""

    @pytest.mark.parametrize("name", STATCAST_PRESET_NAMES)
    def test_statcast_pipeline_produces_projections(self, name: str) -> None:
        """Full pipeline with fake data produces valid projections."""
        league = _make_league_batting()
        statcast_stats = [
            StatcastBatterStats(
                player_id="mlb1",
                name="Test Hitter",
                year=2024,
                pa=500,
                barrel_rate=0.15,
                hard_hit_rate=0.42,
                xwoba=0.360,
                xba=0.270,
                xslg=0.480,
            ),
        ]
        statcast_source = IntegrationStatcastSource({2024: statcast_stats})
        id_mapper = IntegrationIdMapper({"p1": "mlb1"})
        ds = IntegrationDataSource(
            player_batting={
                2024: [_make_batting_stats(year=2024, age=28)],
                2023: [_make_batting_stats(year=2023, age=27)],
                2022: [_make_batting_stats(year=2022, age=26)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        factory = STATCAST_FACTORY_MAP[name]
        pipeline = factory(statcast_source, id_mapper)
        result = pipeline.project_batters(ds, 2025)

        assert len(result) == 1
        assert isinstance(result[0], BattingProjection)
        assert result[0].hr > 0
        assert result[0].pa > 0

    @pytest.mark.parametrize("name", STATCAST_PRESET_NAMES)
    def test_batter_with_statcast_gets_blended(self, name: str) -> None:
        """Batter with Statcast data should have blend metadata."""
        league = _make_league_batting()
        statcast_stats = [
            StatcastBatterStats(
                player_id="mlb1",
                name="Test Hitter",
                year=2024,
                pa=500,
                barrel_rate=0.15,
                hard_hit_rate=0.42,
                xwoba=0.360,
                xba=0.270,
                xslg=0.480,
            ),
        ]
        statcast_source = IntegrationStatcastSource({2024: statcast_stats})
        id_mapper = IntegrationIdMapper({"p1": "mlb1"})
        ds = IntegrationDataSource(
            player_batting={2024: [_make_batting_stats(year=2024, age=28)]},
            team_batting={2024: [league]},
        )

        factory = STATCAST_FACTORY_MAP[name]
        pipeline = factory(statcast_source, id_mapper)
        result = pipeline.project_batters(ds, 2025)
        assert len(result) == 1
        assert result[0].hr > 0

    @pytest.mark.parametrize("name", STATCAST_PRESET_NAMES)
    def test_batter_without_statcast_passes_through(self, name: str) -> None:
        """Batter without Statcast data still gets projections."""
        league = _make_league_batting()
        statcast_source = IntegrationStatcastSource({2024: []})
        id_mapper = IntegrationIdMapper({})
        ds = IntegrationDataSource(
            player_batting={2024: [_make_batting_stats(year=2024, age=28)]},
            team_batting={2024: [league]},
        )

        factory = STATCAST_FACTORY_MAP[name]
        pipeline = factory(statcast_source, id_mapper)
        result = pipeline.project_batters(ds, 2025)
        assert len(result) == 1
        assert result[0].hr > 0

    @pytest.mark.parametrize("name", STATCAST_PRESET_NAMES)
    def test_pitchers_unaffected(self, name: str) -> None:
        """Pitchers should pass through statcast adjuster cleanly."""
        league_b = _make_league_batting()
        league_p = _make_league_pitching()
        statcast_source = IntegrationStatcastSource({2024: []})
        id_mapper = IntegrationIdMapper({})
        ds = IntegrationDataSource(
            player_pitching={2024: [_make_pitching_stats(year=2024, age=28)]},
            team_batting={2024: [league_b]},
            team_pitching={2024: [league_p]},
        )

        factory = STATCAST_FACTORY_MAP[name]
        pipeline = factory(statcast_source, id_mapper)
        result = pipeline.project_pitchers(ds, 2025)
        assert len(result) == 1
        assert result[0].ip > 0


class TestAllPipelinesProduceValidProjections:
    """Every registered pipeline produces non-empty, non-negative batting projections."""

    @pytest.mark.parametrize("name", NON_STATCAST_PRESET_NAMES)
    def test_non_statcast_pipelines_produce_valid_batting(self, name: str) -> None:
        league = _make_league_batting()
        ds = IntegrationDataSource(
            player_batting={
                2024: [_make_batting_stats(year=2024, age=28)],
                2023: [_make_batting_stats(year=2023, age=27)],
                2022: [_make_batting_stats(year=2022, age=26)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        pipeline = PIPELINES[name]()
        result = pipeline.project_batters(ds, 2025)
        assert len(result) == 1
        proj = result[0]
        assert isinstance(proj, BattingProjection)
        assert proj.pa > 0
        assert proj.hr >= 0
        assert proj.h >= 0
        assert proj.bb >= 0

    @pytest.mark.parametrize("name", STATCAST_PRESET_NAMES)
    def test_statcast_pipelines_produce_valid_batting(self, name: str) -> None:
        league = _make_league_batting()
        statcast_stats = [
            StatcastBatterStats(
                player_id="mlb1",
                name="Test Hitter",
                year=2024,
                pa=500,
                barrel_rate=0.15,
                hard_hit_rate=0.42,
                xwoba=0.360,
                xba=0.270,
                xslg=0.480,
            ),
        ]
        statcast_source = IntegrationStatcastSource({2024: statcast_stats})
        id_mapper = IntegrationIdMapper({"p1": "mlb1"})
        ds = IntegrationDataSource(
            player_batting={
                2024: [_make_batting_stats(year=2024, age=28)],
                2023: [_make_batting_stats(year=2023, age=27)],
                2022: [_make_batting_stats(year=2022, age=26)],
            },
            team_batting={2024: [league], 2023: [league], 2022: [league]},
        )

        factory = STATCAST_FACTORY_MAP[name]
        pipeline = factory(statcast_source, id_mapper)
        result = pipeline.project_batters(ds, 2025)
        assert len(result) == 1
        proj = result[0]
        assert isinstance(proj, BattingProjection)
        assert proj.pa > 0
        assert proj.hr >= 0
        assert proj.h >= 0
        assert proj.bb >= 0
