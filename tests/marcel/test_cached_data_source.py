from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore

if TYPE_CHECKING:
    from pathlib import Path
from fantasy_baseball_manager.marcel.data_source import (
    CachedStatsDataSource,
    _deserialize_batting,
    _deserialize_pitching,
    _serialize_batting,
    _serialize_pitching,
)
from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats


def _sample_batting() -> list[BattingSeasonStats]:
    return [
        BattingSeasonStats(
            player_id="1",
            name="Alice",
            year=2023,
            age=28,
            pa=600,
            ab=550,
            h=160,
            singles=100,
            doubles=30,
            triples=5,
            hr=25,
            bb=40,
            so=120,
            hbp=5,
            sf=3,
            sh=2,
            sb=10,
            cs=3,
            r=80,
            rbi=90,
            team="NYY",
        ),
    ]


def _sample_pitching() -> list[PitchingSeasonStats]:
    return [
        PitchingSeasonStats(
            player_id="2",
            name="Bob",
            year=2023,
            age=30,
            ip=180.1,
            g=32,
            gs=32,
            er=60,
            h=150,
            bb=50,
            so=200,
            hr=20,
            hbp=5,
            w=12,
            sv=0,
            hld=0,
            bs=0,
            team="BOS",
        ),
    ]


class TestSerializationRoundTrip:
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


class _FakeDataSource:
    """Records calls to verify cache hit/miss behaviour."""

    def __init__(self) -> None:
        self.call_counts: dict[str, int] = {
            "batting_stats": 0,
            "pitching_stats": 0,
            "team_batting": 0,
            "team_pitching": 0,
        }

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        self.call_counts["batting_stats"] += 1
        return _sample_batting()

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        self.call_counts["pitching_stats"] += 1
        return _sample_pitching()

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        self.call_counts["team_batting"] += 1
        return _sample_batting()

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        self.call_counts["team_pitching"] += 1
        return _sample_pitching()


class TestCachedStatsDataSource:
    def _make_cached(self, tmp_path: Path) -> tuple[CachedStatsDataSource, _FakeDataSource]:
        delegate = _FakeDataSource()
        cache = SqliteCacheStore(tmp_path / "cache.db")
        cached = CachedStatsDataSource(delegate, cache)
        return cached, delegate

    def test_batting_stats_cache_miss_then_hit(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)
        result1 = cached.batting_stats(2023)
        result2 = cached.batting_stats(2023)
        assert result1 == result2 == _sample_batting()
        assert delegate.call_counts["batting_stats"] == 1

    def test_pitching_stats_cache_miss_then_hit(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)
        result1 = cached.pitching_stats(2023)
        result2 = cached.pitching_stats(2023)
        assert result1 == result2 == _sample_pitching()
        assert delegate.call_counts["pitching_stats"] == 1

    def test_team_batting_cache_miss_then_hit(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)
        result1 = cached.team_batting(2023)
        result2 = cached.team_batting(2023)
        assert result1 == result2 == _sample_batting()
        assert delegate.call_counts["team_batting"] == 1

    def test_team_pitching_cache_miss_then_hit(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)
        result1 = cached.team_pitching(2023)
        result2 = cached.team_pitching(2023)
        assert result1 == result2 == _sample_pitching()
        assert delegate.call_counts["team_pitching"] == 1

    def test_different_years_are_cached_separately(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)
        cached.batting_stats(2022)
        cached.batting_stats(2023)
        cached.batting_stats(2022)
        assert delegate.call_counts["batting_stats"] == 2

    def test_different_methods_are_cached_separately(self, tmp_path: Path) -> None:
        cached, delegate = self._make_cached(tmp_path)
        cached.batting_stats(2023)
        cached.team_batting(2023)
        assert delegate.call_counts["batting_stats"] == 1
        assert delegate.call_counts["team_batting"] == 1

    def test_expired_entry_refetches(self, tmp_path: Path) -> None:
        clock_time = 1000.0

        def fake_clock() -> float:
            return clock_time

        delegate = _FakeDataSource()
        cache = SqliteCacheStore(tmp_path / "cache.db", clock=fake_clock)
        cached = CachedStatsDataSource(delegate, cache, ttl_seconds=60)

        cached.batting_stats(2023)
        assert delegate.call_counts["batting_stats"] == 1

        # Advance past TTL
        clock_time = 1061.0
        cached.batting_stats(2023)
        assert delegate.call_counts["batting_stats"] == 2
