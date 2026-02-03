import pytest

from fantasy_baseball_manager.pipeline.statcast_data import (
    CachedStatcastDataSource,
    StatcastBatterStats,
    StatcastPitcherStats,
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


class FakeStatcastDataSource:
    def __init__(
        self,
        batter_data: dict[int, list[StatcastBatterStats]] | None = None,
        pitcher_data: dict[int, list[StatcastPitcherStats]] | None = None,
    ) -> None:
        self._batter_data = batter_data or {}
        self._pitcher_data = pitcher_data or {}
        self.batter_call_count = 0
        self.pitcher_call_count = 0
        # Keep backwards-compatible call_count alias
        self.call_count = 0

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        self.batter_call_count += 1
        self.call_count += 1
        return self._batter_data.get(year, [])

    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]:
        self.pitcher_call_count += 1
        return self._pitcher_data.get(year, [])


SAMPLE_STATS = [
    StatcastBatterStats(
        player_id="545361",
        name="Mike Trout",
        year=2024,
        pa=500,
        barrel_rate=0.18,
        hard_hit_rate=0.45,
        xwoba=0.380,
        xba=0.280,
        xslg=0.520,
    ),
]

SAMPLE_PITCHER_STATS = [
    StatcastPitcherStats(
        player_id="543037",
        name="Gerrit Cole",
        year=2024,
        pa=800,
        xba=0.220,
        xslg=0.360,
        xwoba=0.290,
        xera=3.10,
        barrel_rate=0.07,
        hard_hit_rate=0.32,
    ),
]


class TestStatcastBatterStats:
    def test_creation(self) -> None:
        stats = StatcastBatterStats(
            player_id="123",
            name="Test Player",
            year=2024,
            pa=400,
            barrel_rate=0.10,
            hard_hit_rate=0.40,
            xwoba=0.320,
            xba=0.250,
            xslg=0.420,
        )
        assert stats.player_id == "123"
        assert stats.barrel_rate == 0.10
        assert stats.xba == 0.250

    def test_frozen(self) -> None:
        stats = SAMPLE_STATS[0]
        with pytest.raises(AttributeError):
            stats.pa = 600  # type: ignore[misc]


class TestStatcastPitcherStats:
    def test_creation(self) -> None:
        stats = StatcastPitcherStats(
            player_id="456",
            name="Test Pitcher",
            year=2024,
            pa=700,
            xba=0.230,
            xslg=0.380,
            xwoba=0.300,
            xera=3.50,
            barrel_rate=0.08,
            hard_hit_rate=0.35,
        )
        assert stats.player_id == "456"
        assert stats.xera == 3.50
        assert stats.xba == 0.230
        assert stats.barrel_rate == 0.08

    def test_frozen(self) -> None:
        stats = SAMPLE_PITCHER_STATS[0]
        with pytest.raises(AttributeError):
            stats.pa = 900  # type: ignore[misc]


class TestCachedStatcastDataSource:
    def test_delegates_on_cache_miss(self) -> None:
        delegate = FakeStatcastDataSource(batter_data={2024: SAMPLE_STATS})
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        result = cached.batter_expected_stats(2024)
        assert len(result) == 1
        assert result[0].player_id == "545361"
        assert delegate.call_count == 1

    def test_returns_cached_on_hit(self) -> None:
        delegate = FakeStatcastDataSource(batter_data={2024: SAMPLE_STATS})
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        cached.batter_expected_stats(2024)
        cached.batter_expected_stats(2024)
        assert delegate.call_count == 1

    def test_different_years_not_cached(self) -> None:
        delegate = FakeStatcastDataSource(batter_data={2024: SAMPLE_STATS, 2023: []})
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        cached.batter_expected_stats(2024)
        result = cached.batter_expected_stats(2023)
        assert result == []
        assert delegate.call_count == 2


class TestCachedStatcastDataSourcePitcher:
    def test_pitcher_delegates_on_cache_miss(self) -> None:
        delegate = FakeStatcastDataSource(pitcher_data={2024: SAMPLE_PITCHER_STATS})
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        result = cached.pitcher_expected_stats(2024)
        assert len(result) == 1
        assert result[0].player_id == "543037"
        assert result[0].xera == 3.10
        assert delegate.pitcher_call_count == 1

    def test_pitcher_returns_cached_on_hit(self) -> None:
        delegate = FakeStatcastDataSource(pitcher_data={2024: SAMPLE_PITCHER_STATS})
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        cached.pitcher_expected_stats(2024)
        cached.pitcher_expected_stats(2024)
        assert delegate.pitcher_call_count == 1

    def test_pitcher_different_years_not_cached(self) -> None:
        delegate = FakeStatcastDataSource(pitcher_data={2024: SAMPLE_PITCHER_STATS, 2023: []})
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        cached.pitcher_expected_stats(2024)
        result = cached.pitcher_expected_stats(2023)
        assert result == []
        assert delegate.pitcher_call_count == 2

    def test_pitcher_and_batter_caches_independent(self) -> None:
        delegate = FakeStatcastDataSource(
            batter_data={2024: SAMPLE_STATS},
            pitcher_data={2024: SAMPLE_PITCHER_STATS},
        )
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        batters = cached.batter_expected_stats(2024)
        pitchers = cached.pitcher_expected_stats(2024)
        assert len(batters) == 1
        assert len(pitchers) == 1
        assert batters[0].player_id == "545361"
        assert pitchers[0].player_id == "543037"
