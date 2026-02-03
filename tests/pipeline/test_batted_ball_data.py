import pytest

from fantasy_baseball_manager.pipeline.batted_ball_data import (
    CachedBattedBallDataSource,
    PitcherBattedBallStats,
)


class FakeCacheStore:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], str] = {}

    def get(self, namespace: str, key: str) -> str | None:
        return self._data.get((namespace, key))

    def put(self, namespace: str, key: str, value: str, ttl_seconds: int) -> None:
        self._data[(namespace, key)] = value

    def invalidate(self, namespace: str, key: str | None = None) -> None:
        if key is not None:
            self._data.pop((namespace, key), None)


class FakeBattedBallDelegate:
    def __init__(self, data: dict[int, list[PitcherBattedBallStats]]) -> None:
        self._data = data
        self.call_count = 0

    def pitcher_batted_ball_stats(self, year: int) -> list[PitcherBattedBallStats]:
        self.call_count += 1
        return self._data.get(year, [])


SAMPLE_STATS = [
    PitcherBattedBallStats(
        player_id="12345",
        name="Test Pitcher",
        year=2024,
        pa=700,
        gb_pct=0.48,
        fb_pct=0.30,
        ld_pct=0.22,
        iffb_pct=0.12,
    ),
]


class TestPitcherBattedBallStats:
    def test_creation(self) -> None:
        stats = PitcherBattedBallStats(
            player_id="123",
            name="Test Pitcher",
            year=2024,
            pa=600,
            gb_pct=0.45,
            fb_pct=0.32,
            ld_pct=0.23,
            iffb_pct=0.11,
        )
        assert stats.player_id == "123"
        assert stats.gb_pct == 0.45
        assert stats.iffb_pct == 0.11

    def test_frozen(self) -> None:
        stats = SAMPLE_STATS[0]
        with pytest.raises(AttributeError):
            stats.pa = 900  # type: ignore[misc]


class TestCachedBattedBallDataSource:
    def test_delegates_on_cache_miss(self) -> None:
        delegate = FakeBattedBallDelegate({2024: SAMPLE_STATS})
        cache = FakeCacheStore()
        cached = CachedBattedBallDataSource(delegate, cache, ttl=86400)
        result = cached.pitcher_batted_ball_stats(2024)
        assert len(result) == 1
        assert result[0].player_id == "12345"
        assert delegate.call_count == 1

    def test_returns_cached_on_hit(self) -> None:
        delegate = FakeBattedBallDelegate({2024: SAMPLE_STATS})
        cache = FakeCacheStore()
        cached = CachedBattedBallDataSource(delegate, cache, ttl=86400)
        cached.pitcher_batted_ball_stats(2024)
        cached.pitcher_batted_ball_stats(2024)
        assert delegate.call_count == 1

    def test_different_years_not_cached(self) -> None:
        delegate = FakeBattedBallDelegate({2024: SAMPLE_STATS, 2023: []})
        cache = FakeCacheStore()
        cached = CachedBattedBallDataSource(delegate, cache, ttl=86400)
        cached.pitcher_batted_ball_stats(2024)
        result = cached.pitcher_batted_ball_stats(2023)
        assert result == []
        assert delegate.call_count == 2

    def test_roundtrip_preserves_fields(self) -> None:
        delegate = FakeBattedBallDelegate({2024: SAMPLE_STATS})
        cache = FakeCacheStore()
        cached = CachedBattedBallDataSource(delegate, cache, ttl=86400)
        # First call populates cache
        cached.pitcher_batted_ball_stats(2024)
        # Second call reads from cache
        result = cached.pitcher_batted_ball_stats(2024)
        original = SAMPLE_STATS[0]
        cached_stat = result[0]
        assert cached_stat.player_id == original.player_id
        assert cached_stat.name == original.name
        assert cached_stat.year == original.year
        assert cached_stat.pa == original.pa
        assert cached_stat.gb_pct == original.gb_pct
        assert cached_stat.fb_pct == original.fb_pct
        assert cached_stat.ld_pct == original.ld_pct
        assert cached_stat.iffb_pct == original.iffb_pct
