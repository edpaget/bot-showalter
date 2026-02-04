"""Tests for skill data models and sources."""

import pytest

from fantasy_baseball_manager.pipeline.skill_data import (
    BatterSkillStats,
    CachedSkillDataSource,
    CompositeSkillDataSource,
    PitcherSkillStats,
)


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


class FakePlayerIdMapper:
    """Test double for PlayerIdMapper."""

    def __init__(self, fg_to_mlbam: dict[str, str] | None = None) -> None:
        self._fg_to_mlbam = fg_to_mlbam or {}

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return None

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return None

    def fangraphs_to_mlbam(self, fangraphs_id: str) -> str | None:
        return self._fg_to_mlbam.get(fangraphs_id)

    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
        return None


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
        mapper = FakePlayerIdMapper(fg_to_mlbam={"19755": "545361"})

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
        mapper = FakePlayerIdMapper(fg_to_mlbam={"19755": "545361"})

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
        mapper = FakePlayerIdMapper(fg_to_mlbam={})  # No mapping

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
        mapper = FakePlayerIdMapper()

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
