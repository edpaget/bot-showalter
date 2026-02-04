"""Tests for CachedADPSource."""

from datetime import UTC, datetime

from fantasy_baseball_manager.adp.models import ADPData, ADPEntry
from fantasy_baseball_manager.cache.sources import CachedADPSource


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
        else:
            to_remove = [k for k in self._data if k[0] == namespace]
            for k in to_remove:
                del self._data[k]


class FakeADPSource:
    def __init__(self, adp_data: ADPData) -> None:
        self._adp_data = adp_data
        self.call_count = 0

    def fetch_adp(self) -> ADPData:
        self.call_count += 1
        return self._adp_data


def _sample_adp_data() -> ADPData:
    return ADPData(
        entries=(
            ADPEntry(name="Mike Trout", adp=1.5, positions=("OF",), percent_drafted=99.8),
            ADPEntry(name="Shohei Ohtani", adp=2.3, positions=("DH", "SP"), percent_drafted=99.5),
        ),
        fetched_at=datetime(2025, 3, 15, 12, 0, 0, tzinfo=UTC),
        source="yahoo",
    )


class TestCachedADPSource:
    def test_cache_miss_delegates_and_stores(self) -> None:
        adp_data = _sample_adp_data()
        delegate = FakeADPSource(adp_data)
        cache = FakeCacheStore()
        source = CachedADPSource(delegate, cache, ttl_seconds=86400)

        result = source.fetch_adp()

        assert result == adp_data
        assert delegate.call_count == 1
        assert cache.get("adp_data", "yahoo") is not None

    def test_cache_hit_returns_without_delegating(self) -> None:
        adp_data = _sample_adp_data()
        delegate = FakeADPSource(adp_data)
        cache = FakeCacheStore()
        source = CachedADPSource(delegate, cache, ttl_seconds=86400)

        source.fetch_adp()  # populate cache
        result = source.fetch_adp()  # should hit cache

        assert result == adp_data
        assert delegate.call_count == 1

    def test_round_trip_fidelity(self) -> None:
        adp_data = _sample_adp_data()
        delegate = FakeADPSource(adp_data)
        cache = FakeCacheStore()
        source = CachedADPSource(delegate, cache, ttl_seconds=86400)

        original = source.fetch_adp()
        # Create new source with same cache to test deserialization
        delegate2 = FakeADPSource(ADPData(entries=(), fetched_at=datetime.now(UTC)))
        source2 = CachedADPSource(delegate2, cache, ttl_seconds=86400)
        restored = source2.fetch_adp()

        assert restored == original
        assert len(restored.entries) == 2
        assert restored.entries[0].name == "Mike Trout"
        assert restored.entries[0].positions == ("OF",)
        assert restored.entries[1].positions == ("DH", "SP")
        assert restored.source == "yahoo"

    def test_preserves_fetched_at_timestamp(self) -> None:
        adp_data = _sample_adp_data()
        delegate = FakeADPSource(adp_data)
        cache = FakeCacheStore()
        source = CachedADPSource(delegate, cache, ttl_seconds=86400)

        source.fetch_adp()  # populate cache
        delegate2 = FakeADPSource(ADPData(entries=(), fetched_at=datetime.now(UTC)))
        source2 = CachedADPSource(delegate2, cache, ttl_seconds=86400)
        restored = source2.fetch_adp()

        assert restored.fetched_at == datetime(2025, 3, 15, 12, 0, 0, tzinfo=UTC)

    def test_preserves_percent_drafted(self) -> None:
        adp_data = _sample_adp_data()
        delegate = FakeADPSource(adp_data)
        cache = FakeCacheStore()
        source = CachedADPSource(delegate, cache, ttl_seconds=86400)

        source.fetch_adp()
        delegate2 = FakeADPSource(ADPData(entries=(), fetched_at=datetime.now(UTC)))
        source2 = CachedADPSource(delegate2, cache, ttl_seconds=86400)
        restored = source2.fetch_adp()

        assert restored.entries[0].percent_drafted == 99.8
        assert restored.entries[1].percent_drafted == 99.5

    def test_handles_none_percent_drafted(self) -> None:
        adp_data = ADPData(
            entries=(ADPEntry(name="Player", adp=10.0, positions=("1B",), percent_drafted=None),),
            fetched_at=datetime.now(UTC),
        )
        delegate = FakeADPSource(adp_data)
        cache = FakeCacheStore()
        source = CachedADPSource(delegate, cache, ttl_seconds=86400)

        source.fetch_adp()
        delegate2 = FakeADPSource(ADPData(entries=(), fetched_at=datetime.now(UTC)))
        source2 = CachedADPSource(delegate2, cache, ttl_seconds=86400)
        restored = source2.fetch_adp()

        assert restored.entries[0].percent_drafted is None
