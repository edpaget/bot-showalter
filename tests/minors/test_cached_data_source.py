"""Tests for minors/cached_data_source.py"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore
from fantasy_baseball_manager.minors.cached_data_source import (
    CachedMinorLeagueDataSource,
    _deserialize_batting,
    _deserialize_pitching,
    _serialize_batting,
    _serialize_pitching,
)
from fantasy_baseball_manager.minors.types import (
    MinorLeagueBatterSeasonStats,
    MinorLeagueLevel,
    MinorLeaguePitcherSeasonStats,
)

if TYPE_CHECKING:
    from pathlib import Path


def _sample_batting() -> list[MinorLeagueBatterSeasonStats]:
    return [
        MinorLeagueBatterSeasonStats(
            player_id="12345",
            name="Test Player",
            season=2024,
            age=23,
            level=MinorLeagueLevel.AAA,
            team="Syracuse Mets",
            league="International League",
            pa=500,
            ab=450,
            h=130,
            singles=90,
            doubles=25,
            triples=5,
            hr=10,
            rbi=60,
            r=70,
            bb=40,
            so=100,
            hbp=5,
            sf=5,
            sb=15,
            cs=5,
            avg=0.289,
            obp=0.360,
            slg=0.440,
        ),
    ]


def _sample_pitching() -> list[MinorLeaguePitcherSeasonStats]:
    return [
        MinorLeaguePitcherSeasonStats(
            player_id="54321",
            name="Test Pitcher",
            season=2024,
            age=24,
            level=MinorLeagueLevel.AA,
            team="Binghamton Rumble Ponies",
            league="Eastern League",
            g=25,
            gs=25,
            ip=140.333,
            w=10,
            losses=5,
            sv=0,
            h=120,
            r=55,
            er=50,
            hr=12,
            bb=40,
            so=150,
            hbp=5,
            era=3.21,
            whip=1.14,
        ),
    ]


class TestSerializationRoundTrip:
    """Tests for serialization/deserialization round trips."""

    def test_batting_round_trip(self) -> None:
        original = _sample_batting()
        serialized = _serialize_batting(original)
        deserialized = _deserialize_batting(serialized)
        assert deserialized == original

    def test_pitching_round_trip(self) -> None:
        original = _sample_pitching()
        serialized = _serialize_pitching(original)
        deserialized = _deserialize_pitching(serialized)
        assert deserialized == original

    def test_empty_list_round_trip(self) -> None:
        assert _deserialize_batting(_serialize_batting([])) == []
        assert _deserialize_pitching(_serialize_pitching([])) == []

    def test_multiple_players_round_trip(self) -> None:
        players = [
            MinorLeagueBatterSeasonStats(
                player_id=str(i),
                name=f"Player {i}",
                season=2024,
                age=22 + i,
                level=MinorLeagueLevel.AAA,
                team="Team",
                league="League",
                pa=400 + i * 10,
                ab=350 + i * 10,
                h=100,
                singles=70,
                doubles=20,
                triples=3,
                hr=7,
                rbi=50,
                r=60,
                bb=30,
                so=80,
                hbp=5,
                sf=5,
                sb=10,
                cs=3,
                avg=0.286,
                obp=0.350,
                slg=0.430,
            )
            for i in range(5)
        ]
        serialized = _serialize_batting(players)
        deserialized = _deserialize_batting(serialized)
        assert deserialized == players


class _FakeMinorLeagueDataSource:
    """Fake data source that records calls."""

    def __init__(self) -> None:
        self.call_counts: dict[str, int] = {
            "batting_stats": 0,
            "batting_stats_all_levels": 0,
            "pitching_stats": 0,
            "pitching_stats_all_levels": 0,
        }

    def batting_stats(self, year: int, level: MinorLeagueLevel) -> list[MinorLeagueBatterSeasonStats]:
        self.call_counts["batting_stats"] += 1
        return _sample_batting()

    def batting_stats_all_levels(self, year: int) -> list[MinorLeagueBatterSeasonStats]:
        self.call_counts["batting_stats_all_levels"] += 1
        return _sample_batting()

    def pitching_stats(self, year: int, level: MinorLeagueLevel) -> list[MinorLeaguePitcherSeasonStats]:
        self.call_counts["pitching_stats"] += 1
        return _sample_pitching()

    def pitching_stats_all_levels(self, year: int) -> list[MinorLeaguePitcherSeasonStats]:
        self.call_counts["pitching_stats_all_levels"] += 1
        return _sample_pitching()


class TestCachedMinorLeagueDataSource:
    """Tests for CachedMinorLeagueDataSource."""

    def _make_cached(self, tmp_path: Path) -> tuple[CachedMinorLeagueDataSource, _FakeMinorLeagueDataSource]:
        delegate = _FakeMinorLeagueDataSource()
        cache = SqliteCacheStore(tmp_path / "cache.db")
        # Set current year high so all test years are "historical"
        cached = CachedMinorLeagueDataSource(delegate, cache, current_year=2030)
        return cached, delegate

    def test_batting_stats_cache_miss_then_hit(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)

        result1 = cached.batting_stats(2024, MinorLeagueLevel.AAA)
        result2 = cached.batting_stats(2024, MinorLeagueLevel.AAA)

        assert result1 == result2 == _sample_batting()
        assert delegate.call_counts["batting_stats"] == 1

    def test_pitching_stats_cache_miss_then_hit(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)

        result1 = cached.pitching_stats(2024, MinorLeagueLevel.AA)
        result2 = cached.pitching_stats(2024, MinorLeagueLevel.AA)

        assert result1 == result2 == _sample_pitching()
        assert delegate.call_counts["pitching_stats"] == 1

    def test_batting_stats_all_levels_cache_miss_then_hit(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)

        result1 = cached.batting_stats_all_levels(2024)
        result2 = cached.batting_stats_all_levels(2024)

        assert result1 == result2 == _sample_batting()
        assert delegate.call_counts["batting_stats_all_levels"] == 1

    def test_pitching_stats_all_levels_cache_miss_then_hit(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)

        result1 = cached.pitching_stats_all_levels(2024)
        result2 = cached.pitching_stats_all_levels(2024)

        assert result1 == result2 == _sample_pitching()
        assert delegate.call_counts["pitching_stats_all_levels"] == 1

    def test_different_years_cached_separately(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)

        cached.batting_stats(2023, MinorLeagueLevel.AAA)
        cached.batting_stats(2024, MinorLeagueLevel.AAA)
        cached.batting_stats(2023, MinorLeagueLevel.AAA)

        assert delegate.call_counts["batting_stats"] == 2

    def test_different_levels_cached_separately(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)

        cached.batting_stats(2024, MinorLeagueLevel.AAA)
        cached.batting_stats(2024, MinorLeagueLevel.AA)
        cached.batting_stats(2024, MinorLeagueLevel.AAA)

        assert delegate.call_counts["batting_stats"] == 2

    def test_batting_and_pitching_cached_separately(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)

        cached.batting_stats(2024, MinorLeagueLevel.AAA)
        cached.pitching_stats(2024, MinorLeagueLevel.AAA)

        assert delegate.call_counts["batting_stats"] == 1
        assert delegate.call_counts["pitching_stats"] == 1

    def test_current_year_uses_shorter_ttl(self, tmp_path: Path) -> None:
        """Current year data should be cached with 1 day TTL."""
        clock_time = 1000.0

        def fake_clock() -> float:
            return clock_time

        delegate = _FakeMinorLeagueDataSource()
        cache = SqliteCacheStore(tmp_path / "cache.db", clock=fake_clock)
        # Set current year to match test year
        cached = CachedMinorLeagueDataSource(delegate, cache, current_year=2024)

        # First call - cache miss
        cached.batting_stats(2024, MinorLeagueLevel.AAA)
        assert delegate.call_counts["batting_stats"] == 1

        # Within TTL (1 day = 86400 seconds) - cache hit
        clock_time = 1000.0 + 3600  # 1 hour later
        cached.batting_stats(2024, MinorLeagueLevel.AAA)
        assert delegate.call_counts["batting_stats"] == 1

        # Past TTL - cache miss
        clock_time = 1000.0 + 86401  # 1 day + 1 second
        cached.batting_stats(2024, MinorLeagueLevel.AAA)
        assert delegate.call_counts["batting_stats"] == 2

    def test_historical_year_uses_longer_ttl(self, tmp_path: Path) -> None:
        """Historical year data should be cached with 1 year TTL."""
        clock_time = 1000.0

        def fake_clock() -> float:
            return clock_time

        delegate = _FakeMinorLeagueDataSource()
        cache = SqliteCacheStore(tmp_path / "cache.db", clock=fake_clock)
        # Set current year after test year
        cached = CachedMinorLeagueDataSource(delegate, cache, current_year=2025)

        # First call - cache miss
        cached.batting_stats(2024, MinorLeagueLevel.AAA)
        assert delegate.call_counts["batting_stats"] == 1

        # 30 days later - still cached (historical data)
        clock_time = 1000.0 + 86400 * 30
        cached.batting_stats(2024, MinorLeagueLevel.AAA)
        assert delegate.call_counts["batting_stats"] == 1

        # 1 year + 1 day later - cache expired
        clock_time = 1000.0 + 86400 * 366
        cached.batting_stats(2024, MinorLeagueLevel.AAA)
        assert delegate.call_counts["batting_stats"] == 2
