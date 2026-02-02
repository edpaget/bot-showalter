from fantasy_baseball_manager.pipeline.statcast_data import (
    CachedStatcastDataSource,
    StatcastBatterStats,
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
    def __init__(self, data: dict[int, list[StatcastBatterStats]]) -> None:
        self._data = data
        self.call_count = 0

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        self.call_count += 1
        return self._data.get(year, [])


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
        import pytest

        stats = SAMPLE_STATS[0]
        with pytest.raises(AttributeError):
            stats.pa = 600  # type: ignore[misc]


class TestCachedStatcastDataSource:
    def test_delegates_on_cache_miss(self) -> None:
        delegate = FakeStatcastDataSource({2024: SAMPLE_STATS})
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        result = cached.batter_expected_stats(2024)
        assert len(result) == 1
        assert result[0].player_id == "545361"
        assert delegate.call_count == 1

    def test_returns_cached_on_hit(self) -> None:
        delegate = FakeStatcastDataSource({2024: SAMPLE_STATS})
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        cached.batter_expected_stats(2024)
        cached.batter_expected_stats(2024)
        assert delegate.call_count == 1

    def test_different_years_not_cached(self) -> None:
        delegate = FakeStatcastDataSource({2024: SAMPLE_STATS, 2023: []})
        cache = FakeCacheStore()
        cached = CachedStatcastDataSource(delegate, cache, ttl=86400)
        cached.batter_expected_stats(2024)
        result = cached.batter_expected_stats(2023)
        assert result == []
        assert delegate.call_count == 2
