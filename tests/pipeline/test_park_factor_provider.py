import pytest

from fantasy_baseball_manager.pipeline.park_factors import (
    CachedParkFactorProvider,
    FanGraphsParkFactorProvider,
)


class FakeParkFactorDelegate:
    def __init__(self, data: dict[str, dict[str, float]]) -> None:
        self._data = data
        self.call_count = 0

    def park_factors(self, year: int) -> dict[str, dict[str, float]]:
        self.call_count += 1
        return self._data


class FakeCacheStore:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get(self, namespace: str, key: str) -> str | None:
        return self._store.get((namespace, key))

    def put(self, namespace: str, key: str, value: str, ttl_seconds: int) -> None:
        self._store[(namespace, key)] = value

    def invalidate(self, namespace: str, key: str | None = None) -> None:
        if key is not None:
            self._store.pop((namespace, key), None)
        else:
            to_delete = [k for k in self._store if k[0] == namespace]
            for k in to_delete:
                del self._store[k]


class TestFanGraphsParkFactorProvider:
    def test_regression_toward_one(self) -> None:
        """With regression_weight=0.5, a raw factor of 1.2 becomes 1.1."""
        provider = FanGraphsParkFactorProvider(regression_weight=0.5)
        result = provider._regress(1.2)
        assert result == pytest.approx(1.1)

    def test_regression_weight_zero_returns_one(self) -> None:
        """With regression_weight=0.0, all factors become 1.0."""
        provider = FanGraphsParkFactorProvider(regression_weight=0.0)
        result = provider._regress(1.5)
        assert result == pytest.approx(1.0)

    def test_regression_weight_one_returns_raw(self) -> None:
        """With regression_weight=1.0, raw factor is returned unchanged."""
        provider = FanGraphsParkFactorProvider(regression_weight=1.0)
        result = provider._regress(1.3)
        assert result == pytest.approx(1.3)

    def test_column_map_has_expected_stats(self) -> None:
        expected = {"hr", "singles", "doubles", "triples", "bb", "so"}
        assert set(FanGraphsParkFactorProvider._column_map().values()) == expected


class TestCachedParkFactorProvider:
    def test_caches_result(self) -> None:
        data = {"COL": {"hr": 1.07}}
        delegate = FakeParkFactorDelegate(data)
        cache = FakeCacheStore()
        provider = CachedParkFactorProvider(delegate, cache)

        result1 = provider.park_factors(2023)
        result2 = provider.park_factors(2023)

        assert result1 == data
        assert result2 == data
        assert delegate.call_count == 1

    def test_cache_miss_calls_delegate(self) -> None:
        data = {"NYY": {"hr": 1.04}}
        delegate = FakeParkFactorDelegate(data)
        cache = FakeCacheStore()
        provider = CachedParkFactorProvider(delegate, cache)

        result = provider.park_factors(2023)
        assert result == data
        assert delegate.call_count == 1

    def test_different_years_cached_separately(self) -> None:
        data = {"COL": {"hr": 1.07}}
        delegate = FakeParkFactorDelegate(data)
        cache = FakeCacheStore()
        provider = CachedParkFactorProvider(delegate, cache)

        provider.park_factors(2022)
        provider.park_factors(2023)
        provider.park_factors(2022)

        assert delegate.call_count == 2
