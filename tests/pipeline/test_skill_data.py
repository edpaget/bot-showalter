"""Tests for skill data models and sources."""

import pytest

from fantasy_baseball_manager.pipeline.skill_data import (
    BatterSkillDelta,
    BatterSkillStats,
    CachedSkillDataSource,
    CompositeSkillDataSource,
    PitcherSkillDelta,
    PitcherSkillStats,
    SkillDeltaComputer,
)
from fantasy_baseball_manager.player_id.mapper import SfbbMapper
from tests.conftest import make_test_feature_store


class FakeCacheStore:
    """Test double for CacheStore."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def get(self, namespace: str, key: str) -> str | None:
        return self._data.get((namespace, key))

    def put(self, namespace: str, key: str, value: str, ttl_seconds: int) -> None:
        self._data[(namespace, key)] = value

    def invalidate(self, namespace: str, key: str | None = None) -> None:
        if key is not None:
            self._data.pop((namespace, key), None)


class FakeSkillDataSource:
    """Test double for SkillDataSource."""

    def __init__(
        self,
        batter_data: dict[int, list[BatterSkillStats]] | None = None,
        pitcher_data: dict[int, list[PitcherSkillStats]] | None = None,
    ) -> None:
        self._batter_data = batter_data or {}
        self._pitcher_data = pitcher_data or {}
        self.batter_call_count = 0
        self.pitcher_call_count = 0

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]:
        self.batter_call_count += 1
        return self._batter_data.get(year, [])

    def pitcher_skill_stats(self, year: int) -> list[PitcherSkillStats]:
        self.pitcher_call_count += 1
        return self._pitcher_data.get(year, [])


def _fake_mapper(fg_to_mlbam: dict[str, str] | None = None) -> SfbbMapper:
    """Create an SfbbMapper with optional fg->mlbam mappings."""
    return SfbbMapper({}, {}, fg_to_mlbam=fg_to_mlbam)


class FakeFanGraphsSkillDataSource:
    """Test double for FanGraphsSkillDataSource."""

    def __init__(self, batters: list[BatterSkillStats], pitchers: list[PitcherSkillStats]) -> None:
        self._batters = batters
        self._pitchers = pitchers

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]:
        return [b for b in self._batters if b.year == year]

    def pitcher_skill_stats(self, year: int) -> list[PitcherSkillStats]:
        return [p for p in self._pitchers if p.year == year]


class FakeSprintSpeedSource:
    """Test double for StatcastSprintSpeedSource."""

    def __init__(self, speeds: dict[int, dict[str, float]]) -> None:
        self._speeds = speeds  # year -> (mlbam_id -> speed)

    def sprint_speeds(self, year: int) -> dict[str, float]:
        return self._speeds.get(year, {})


SAMPLE_BATTER = BatterSkillStats(
    player_id="19755",  # FanGraphs ID for Mike Trout
    name="Mike Trout",
    year=2024,
    pa=500,
    barrel_rate=0.18,
    hard_hit_rate=0.45,
    exit_velo_avg=93.5,
    exit_velo_max=115.2,
    chase_rate=0.25,
    whiff_rate=0.12,
    sprint_speed=29.5,
)

SAMPLE_BATTER_NO_SPRINT = BatterSkillStats(
    player_id="19755",
    name="Mike Trout",
    year=2024,
    pa=500,
    barrel_rate=0.18,
    hard_hit_rate=0.45,
    exit_velo_avg=93.5,
    exit_velo_max=115.2,
    chase_rate=0.25,
    whiff_rate=0.12,
    sprint_speed=None,
)

SAMPLE_PITCHER = PitcherSkillStats(
    player_id="13125",  # FanGraphs ID for Gerrit Cole
    name="Gerrit Cole",
    year=2024,
    pa_against=800,
    fastball_velo=97.5,
    whiff_rate=0.14,
    gb_rate=0.42,
    barrel_rate_against=0.07,
)


class TestBatterSkillStats:
    def test_dataclass_fields(self) -> None:
        stats = SAMPLE_BATTER
        assert stats.player_id == "19755"
        assert stats.name == "Mike Trout"
        assert stats.year == 2024
        assert stats.pa == 500
        assert stats.barrel_rate == 0.18
        assert stats.hard_hit_rate == 0.45
        assert stats.exit_velo_avg == 93.5
        assert stats.exit_velo_max == 115.2
        assert stats.chase_rate == 0.25
        assert stats.whiff_rate == 0.12
        assert stats.sprint_speed == 29.5

    def test_sprint_speed_optional(self) -> None:
        stats = BatterSkillStats(
            player_id="123",
            name="Test Player",
            year=2024,
            pa=400,
            barrel_rate=0.10,
            hard_hit_rate=0.40,
            exit_velo_avg=90.0,
            exit_velo_max=110.0,
            chase_rate=0.30,
            whiff_rate=0.10,
            sprint_speed=None,
        )
        assert stats.sprint_speed is None

    def test_frozen(self) -> None:
        stats = SAMPLE_BATTER
        with pytest.raises(AttributeError):
            stats.pa = 600  # type: ignore[misc]


class TestPitcherSkillStats:
    def test_dataclass_fields(self) -> None:
        stats = SAMPLE_PITCHER
        assert stats.player_id == "13125"
        assert stats.name == "Gerrit Cole"
        assert stats.year == 2024
        assert stats.pa_against == 800
        assert stats.fastball_velo == 97.5
        assert stats.whiff_rate == 0.14
        assert stats.gb_rate == 0.42
        assert stats.barrel_rate_against == 0.07

    def test_optional_fields(self) -> None:
        stats = PitcherSkillStats(
            player_id="456",
            name="R.A. Dickey",
            year=2024,
            pa_against=600,
            fastball_velo=None,  # Knuckleballer
            whiff_rate=0.08,
            gb_rate=0.50,
            barrel_rate_against=None,
        )
        assert stats.fastball_velo is None
        assert stats.barrel_rate_against is None

    def test_frozen(self) -> None:
        stats = SAMPLE_PITCHER
        with pytest.raises(AttributeError):
            stats.pa_against = 900  # type: ignore[misc]


class TestCompositeSkillDataSource:
    def test_merges_sprint_speed(self) -> None:
        fangraphs = FakeFanGraphsSkillDataSource(
            batters=[SAMPLE_BATTER_NO_SPRINT],
            pitchers=[],
        )
        sprint = FakeSprintSpeedSource({2024: {"545361": 29.5}})  # MLBAM ID
        mapper = _fake_mapper(fg_to_mlbam={"19755": "545361"})

        source = CompositeSkillDataSource(fangraphs, sprint, mapper)
        result = source.batter_skill_stats(2024)

        assert len(result) == 1
        assert result[0].player_id == "19755"
        assert result[0].sprint_speed == 29.5

    def test_handles_missing_sprint_speed(self) -> None:
        fangraphs = FakeFanGraphsSkillDataSource(
            batters=[SAMPLE_BATTER_NO_SPRINT],
            pitchers=[],
        )
        sprint = FakeSprintSpeedSource({2024: {}})  # No sprint data
        mapper = _fake_mapper(fg_to_mlbam={"19755": "545361"})

        source = CompositeSkillDataSource(fangraphs, sprint, mapper)
        result = source.batter_skill_stats(2024)

        assert len(result) == 1
        assert result[0].sprint_speed is None

    def test_handles_missing_id_mapping(self) -> None:
        fangraphs = FakeFanGraphsSkillDataSource(
            batters=[SAMPLE_BATTER_NO_SPRINT],
            pitchers=[],
        )
        sprint = FakeSprintSpeedSource({2024: {"545361": 29.5}})
        mapper = _fake_mapper(fg_to_mlbam={})  # No mapping

        source = CompositeSkillDataSource(fangraphs, sprint, mapper)
        result = source.batter_skill_stats(2024)

        assert len(result) == 1
        assert result[0].sprint_speed is None

    def test_pitcher_stats_passthrough(self) -> None:
        fangraphs = FakeFanGraphsSkillDataSource(
            batters=[],
            pitchers=[SAMPLE_PITCHER],
        )
        sprint = FakeSprintSpeedSource({})
        mapper = _fake_mapper()

        source = CompositeSkillDataSource(fangraphs, sprint, mapper)
        result = source.pitcher_skill_stats(2024)

        assert len(result) == 1
        assert result[0].player_id == "13125"
        assert result[0].fastball_velo == 97.5


class TestCachedSkillDataSource:
    def test_batter_delegates_on_cache_miss(self) -> None:
        delegate = FakeSkillDataSource(batter_data={2024: [SAMPLE_BATTER]})
        cache = FakeCacheStore()
        cached = CachedSkillDataSource(delegate, cache, ttl=86400)

        result = cached.batter_skill_stats(2024)

        assert len(result) == 1
        assert result[0].player_id == "19755"
        assert delegate.batter_call_count == 1

    def test_batter_returns_cached_on_hit(self) -> None:
        delegate = FakeSkillDataSource(batter_data={2024: [SAMPLE_BATTER]})
        cache = FakeCacheStore()
        cached = CachedSkillDataSource(delegate, cache, ttl=86400)

        cached.batter_skill_stats(2024)
        cached.batter_skill_stats(2024)

        assert delegate.batter_call_count == 1

    def test_batter_different_years_not_cached(self) -> None:
        delegate = FakeSkillDataSource(batter_data={2024: [SAMPLE_BATTER], 2023: []})
        cache = FakeCacheStore()
        cached = CachedSkillDataSource(delegate, cache, ttl=86400)

        cached.batter_skill_stats(2024)
        result = cached.batter_skill_stats(2023)

        assert result == []
        assert delegate.batter_call_count == 2

    def test_pitcher_delegates_on_cache_miss(self) -> None:
        delegate = FakeSkillDataSource(pitcher_data={2024: [SAMPLE_PITCHER]})
        cache = FakeCacheStore()
        cached = CachedSkillDataSource(delegate, cache, ttl=86400)

        result = cached.pitcher_skill_stats(2024)

        assert len(result) == 1
        assert result[0].player_id == "13125"
        assert delegate.pitcher_call_count == 1

    def test_pitcher_returns_cached_on_hit(self) -> None:
        delegate = FakeSkillDataSource(pitcher_data={2024: [SAMPLE_PITCHER]})
        cache = FakeCacheStore()
        cached = CachedSkillDataSource(delegate, cache, ttl=86400)

        cached.pitcher_skill_stats(2024)
        cached.pitcher_skill_stats(2024)

        assert delegate.pitcher_call_count == 1

    def test_batter_and_pitcher_caches_independent(self) -> None:
        delegate = FakeSkillDataSource(
            batter_data={2024: [SAMPLE_BATTER]},
            pitcher_data={2024: [SAMPLE_PITCHER]},
        )
        cache = FakeCacheStore()
        cached = CachedSkillDataSource(delegate, cache, ttl=86400)

        batters = cached.batter_skill_stats(2024)
        pitchers = cached.pitcher_skill_stats(2024)

        assert len(batters) == 1
        assert len(pitchers) == 1
        assert batters[0].player_id == "19755"
        assert pitchers[0].player_id == "13125"


# Sample data for delta tests - two years of data for the same player
BATTER_2023 = BatterSkillStats(
    player_id="19755",
    name="Mike Trout",
    year=2023,
    pa=450,
    barrel_rate=0.16,
    hard_hit_rate=0.42,
    exit_velo_avg=92.0,
    exit_velo_max=113.5,
    chase_rate=0.28,
    whiff_rate=0.14,
    sprint_speed=28.5,
)

BATTER_2024 = BatterSkillStats(
    player_id="19755",
    name="Mike Trout",
    year=2024,
    pa=500,
    barrel_rate=0.18,
    hard_hit_rate=0.45,
    exit_velo_avg=93.5,
    exit_velo_max=115.2,
    chase_rate=0.25,
    whiff_rate=0.12,
    sprint_speed=29.5,
)

PITCHER_2023 = PitcherSkillStats(
    player_id="13125",
    name="Gerrit Cole",
    year=2023,
    pa_against=750,
    fastball_velo=96.5,
    whiff_rate=0.13,
    gb_rate=0.40,
    barrel_rate_against=0.08,
)

PITCHER_2024 = PitcherSkillStats(
    player_id="13125",
    name="Gerrit Cole",
    year=2024,
    pa_against=800,
    fastball_velo=97.5,
    whiff_rate=0.14,
    gb_rate=0.42,
    barrel_rate_against=0.07,
)


class TestBatterSkillDelta:
    def test_dataclass_fields(self) -> None:
        delta = BatterSkillDelta(
            player_id="19755",
            name="Mike Trout",
            year=2025,
            barrel_rate_delta=0.02,
            hard_hit_rate_delta=0.03,
            exit_velo_avg_delta=1.5,
            exit_velo_max_delta=1.7,
            chase_rate_delta=-0.03,
            whiff_rate_delta=-0.02,
            sprint_speed_delta=1.0,
            pa_current=500,
            pa_prior=450,
        )
        assert delta.player_id == "19755"
        assert delta.year == 2025
        assert delta.barrel_rate_delta == 0.02
        assert delta.chase_rate_delta == -0.03
        assert delta.pa_current == 500
        assert delta.pa_prior == 450

    def test_has_sufficient_sample_both_above(self) -> None:
        delta = BatterSkillDelta(
            player_id="123",
            name="Test",
            year=2025,
            barrel_rate_delta=0.0,
            hard_hit_rate_delta=0.0,
            exit_velo_avg_delta=0.0,
            exit_velo_max_delta=0.0,
            chase_rate_delta=0.0,
            whiff_rate_delta=0.0,
            sprint_speed_delta=0.0,
            pa_current=250,
            pa_prior=220,
        )
        assert delta.has_sufficient_sample(min_pa=200)

    def test_has_sufficient_sample_current_below(self) -> None:
        delta = BatterSkillDelta(
            player_id="123",
            name="Test",
            year=2025,
            barrel_rate_delta=0.0,
            hard_hit_rate_delta=0.0,
            exit_velo_avg_delta=0.0,
            exit_velo_max_delta=0.0,
            chase_rate_delta=0.0,
            whiff_rate_delta=0.0,
            sprint_speed_delta=0.0,
            pa_current=150,
            pa_prior=250,
        )
        assert not delta.has_sufficient_sample(min_pa=200)

    def test_has_sufficient_sample_prior_below(self) -> None:
        delta = BatterSkillDelta(
            player_id="123",
            name="Test",
            year=2025,
            barrel_rate_delta=0.0,
            hard_hit_rate_delta=0.0,
            exit_velo_avg_delta=0.0,
            exit_velo_max_delta=0.0,
            chase_rate_delta=0.0,
            whiff_rate_delta=0.0,
            sprint_speed_delta=0.0,
            pa_current=250,
            pa_prior=150,
        )
        assert not delta.has_sufficient_sample(min_pa=200)

    def test_frozen(self) -> None:
        delta = BatterSkillDelta(
            player_id="123",
            name="Test",
            year=2025,
            barrel_rate_delta=0.0,
            hard_hit_rate_delta=0.0,
            exit_velo_avg_delta=0.0,
            exit_velo_max_delta=0.0,
            chase_rate_delta=0.0,
            whiff_rate_delta=0.0,
            sprint_speed_delta=0.0,
            pa_current=250,
            pa_prior=250,
        )
        with pytest.raises(AttributeError):
            delta.pa_current = 300  # type: ignore[misc]


class TestPitcherSkillDelta:
    def test_dataclass_fields(self) -> None:
        delta = PitcherSkillDelta(
            player_id="13125",
            name="Gerrit Cole",
            year=2025,
            fastball_velo_delta=1.0,
            whiff_rate_delta=0.01,
            gb_rate_delta=0.02,
            barrel_rate_against_delta=-0.01,
            pa_against_current=800,
            pa_against_prior=750,
        )
        assert delta.player_id == "13125"
        assert delta.fastball_velo_delta == 1.0
        assert delta.barrel_rate_against_delta == -0.01

    def test_has_sufficient_sample(self) -> None:
        delta = PitcherSkillDelta(
            player_id="123",
            name="Test",
            year=2025,
            fastball_velo_delta=0.0,
            whiff_rate_delta=0.0,
            gb_rate_delta=0.0,
            barrel_rate_against_delta=0.0,
            pa_against_current=300,
            pa_against_prior=280,
        )
        assert delta.has_sufficient_sample(min_pa=200)

    def test_has_insufficient_sample(self) -> None:
        delta = PitcherSkillDelta(
            player_id="123",
            name="Test",
            year=2025,
            fastball_velo_delta=0.0,
            whiff_rate_delta=0.0,
            gb_rate_delta=0.0,
            barrel_rate_against_delta=0.0,
            pa_against_current=100,
            pa_against_prior=280,
        )
        assert not delta.has_sufficient_sample(min_pa=200)

    def test_optional_fields_none(self) -> None:
        delta = PitcherSkillDelta(
            player_id="123",
            name="Test",
            year=2025,
            fastball_velo_delta=None,
            whiff_rate_delta=0.0,
            gb_rate_delta=0.0,
            barrel_rate_against_delta=None,
            pa_against_current=300,
            pa_against_prior=280,
        )
        assert delta.fastball_velo_delta is None
        assert delta.barrel_rate_against_delta is None


class TestSkillDeltaComputer:
    def test_compute_batter_deltas(self) -> None:
        source = FakeSkillDataSource(
            batter_data={
                2023: [BATTER_2023],
                2024: [BATTER_2024],
            }
        )
        computer = SkillDeltaComputer(feature_store=make_test_feature_store(skill_data_source=source))

        # Year 2025 means we compare 2023 (year-2) to 2024 (year-1)
        deltas = computer.compute_batter_deltas(2025)

        assert len(deltas) == 1
        assert "19755" in deltas

        delta = deltas["19755"]
        assert delta.player_id == "19755"
        assert delta.name == "Mike Trout"
        assert delta.year == 2025
        assert delta.barrel_rate_delta == pytest.approx(0.02)  # 0.18 - 0.16
        assert delta.hard_hit_rate_delta == pytest.approx(0.03)  # 0.45 - 0.42
        assert delta.exit_velo_avg_delta == pytest.approx(1.5)  # 93.5 - 92.0
        assert delta.exit_velo_max_delta == pytest.approx(1.7)  # 115.2 - 113.5
        assert delta.chase_rate_delta == pytest.approx(-0.03)  # 0.25 - 0.28
        assert delta.whiff_rate_delta == pytest.approx(-0.02)  # 0.12 - 0.14
        assert delta.sprint_speed_delta == pytest.approx(1.0)  # 29.5 - 28.5
        assert delta.pa_current == 500
        assert delta.pa_prior == 450

    def test_compute_batter_deltas_missing_prior_year(self) -> None:
        source = FakeSkillDataSource(
            batter_data={
                2024: [BATTER_2024],
                # 2023 missing
            }
        )
        computer = SkillDeltaComputer(feature_store=make_test_feature_store(skill_data_source=source))

        deltas = computer.compute_batter_deltas(2025)

        assert len(deltas) == 0

    def test_compute_batter_deltas_missing_current_year(self) -> None:
        source = FakeSkillDataSource(
            batter_data={
                2023: [BATTER_2023],
                # 2024 missing
            }
        )
        computer = SkillDeltaComputer(feature_store=make_test_feature_store(skill_data_source=source))

        deltas = computer.compute_batter_deltas(2025)

        assert len(deltas) == 0

    def test_compute_batter_deltas_different_players(self) -> None:
        batter_a_2023 = BatterSkillStats(
            player_id="111",
            name="Player A",
            year=2023,
            pa=400,
            barrel_rate=0.10,
            hard_hit_rate=0.40,
            exit_velo_avg=90.0,
            exit_velo_max=110.0,
            chase_rate=0.30,
            whiff_rate=0.15,
            sprint_speed=27.0,
        )
        batter_b_2024 = BatterSkillStats(
            player_id="222",
            name="Player B",
            year=2024,
            pa=400,
            barrel_rate=0.12,
            hard_hit_rate=0.42,
            exit_velo_avg=91.0,
            exit_velo_max=111.0,
            chase_rate=0.28,
            whiff_rate=0.13,
            sprint_speed=28.0,
        )
        source = FakeSkillDataSource(
            batter_data={
                2023: [batter_a_2023],
                2024: [batter_b_2024],
            }
        )
        computer = SkillDeltaComputer(feature_store=make_test_feature_store(skill_data_source=source))

        deltas = computer.compute_batter_deltas(2025)

        # No overlap in player IDs
        assert len(deltas) == 0

    def test_compute_batter_deltas_sprint_speed_none(self) -> None:
        batter_2023_no_sprint = BatterSkillStats(
            player_id="19755",
            name="Mike Trout",
            year=2023,
            pa=450,
            barrel_rate=0.16,
            hard_hit_rate=0.42,
            exit_velo_avg=92.0,
            exit_velo_max=113.5,
            chase_rate=0.28,
            whiff_rate=0.14,
            sprint_speed=None,
        )
        source = FakeSkillDataSource(
            batter_data={
                2023: [batter_2023_no_sprint],
                2024: [BATTER_2024],
            }
        )
        computer = SkillDeltaComputer(feature_store=make_test_feature_store(skill_data_source=source))

        deltas = computer.compute_batter_deltas(2025)

        assert len(deltas) == 1
        delta = deltas["19755"]
        # Sprint speed delta should be None when prior year is None
        assert delta.sprint_speed_delta is None
        # Other deltas should still be computed
        assert delta.barrel_rate_delta == pytest.approx(0.02)

    def test_compute_pitcher_deltas(self) -> None:
        source = FakeSkillDataSource(
            pitcher_data={
                2023: [PITCHER_2023],
                2024: [PITCHER_2024],
            }
        )
        computer = SkillDeltaComputer(feature_store=make_test_feature_store(skill_data_source=source))

        deltas = computer.compute_pitcher_deltas(2025)

        assert len(deltas) == 1
        assert "13125" in deltas

        delta = deltas["13125"]
        assert delta.player_id == "13125"
        assert delta.name == "Gerrit Cole"
        assert delta.year == 2025
        assert delta.fastball_velo_delta == pytest.approx(1.0)  # 97.5 - 96.5
        assert delta.whiff_rate_delta == pytest.approx(0.01)  # 0.14 - 0.13
        assert delta.gb_rate_delta == pytest.approx(0.02)  # 0.42 - 0.40
        assert delta.barrel_rate_against_delta == pytest.approx(-0.01)  # 0.07 - 0.08
        assert delta.pa_against_current == 800
        assert delta.pa_against_prior == 750

    def test_compute_pitcher_deltas_velo_none(self) -> None:
        pitcher_no_velo = PitcherSkillStats(
            player_id="99999",
            name="R.A. Dickey",
            year=2023,
            pa_against=600,
            fastball_velo=None,
            whiff_rate=0.08,
            gb_rate=0.50,
            barrel_rate_against=None,
        )
        pitcher_no_velo_2024 = PitcherSkillStats(
            player_id="99999",
            name="R.A. Dickey",
            year=2024,
            pa_against=580,
            fastball_velo=None,
            whiff_rate=0.09,
            gb_rate=0.48,
            barrel_rate_against=None,
        )
        source = FakeSkillDataSource(
            pitcher_data={
                2023: [pitcher_no_velo],
                2024: [pitcher_no_velo_2024],
            }
        )
        computer = SkillDeltaComputer(feature_store=make_test_feature_store(skill_data_source=source))

        deltas = computer.compute_pitcher_deltas(2025)

        assert len(deltas) == 1
        delta = deltas["99999"]
        assert delta.fastball_velo_delta is None
        assert delta.barrel_rate_against_delta is None
        assert delta.whiff_rate_delta == pytest.approx(0.01)
        assert delta.gb_rate_delta == pytest.approx(-0.02)

    def test_compute_pitcher_deltas_missing_year(self) -> None:
        source = FakeSkillDataSource(
            pitcher_data={
                2024: [PITCHER_2024],
            }
        )
        computer = SkillDeltaComputer(feature_store=make_test_feature_store(skill_data_source=source))

        deltas = computer.compute_pitcher_deltas(2025)

        assert len(deltas) == 0

