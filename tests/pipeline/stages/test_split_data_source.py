from fantasy_baseball_manager.marcel.models import BattingSeasonStats
from fantasy_baseball_manager.pipeline.stages.split_data_source import (
    CachedSplitDataSource,
    SplitStatsDataSource,
)


def _make_batting(
    player_id: str = "p1",
    name: str = "Hitter",
    year: int = 2024,
    pa: int = 300,
) -> BattingSeasonStats:
    return BattingSeasonStats(
        player_id=player_id,
        name=name,
        year=year,
        age=28,
        pa=pa,
        ab=270,
        h=80,
        singles=50,
        doubles=15,
        triples=2,
        hr=13,
        bb=25,
        so=60,
        hbp=3,
        sf=2,
        sh=0,
        sb=5,
        cs=2,
        r=40,
        rbi=45,
    )


class FakeSplitSource:
    """Implements SplitStatsDataSource protocol for testing."""

    def __init__(
        self,
        vs_lhp: dict[int, list[BattingSeasonStats]] | None = None,
        vs_rhp: dict[int, list[BattingSeasonStats]] | None = None,
    ) -> None:
        self._vs_lhp = vs_lhp or {}
        self._vs_rhp = vs_rhp or {}

    def batting_stats_vs_lhp(self, year: int) -> list[BattingSeasonStats]:
        return self._vs_lhp.get(year, [])

    def batting_stats_vs_rhp(self, year: int) -> list[BattingSeasonStats]:
        return self._vs_rhp.get(year, [])


class FakeCacheStore:
    """In-memory cache for testing."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, str]] = {}

    def get(self, namespace: str, key: str) -> str | None:
        return self._store.get(namespace, {}).get(key)

    def put(self, namespace: str, key: str, value: str, ttl_seconds: int) -> None:
        self._store.setdefault(namespace, {})[key] = value

    def invalidate(self, namespace: str, key: str | None = None) -> None:
        if key is None:
            self._store.pop(namespace, None)
        else:
            self._store.get(namespace, {}).pop(key, None)


class TestSplitStatsDataSourceProtocol:
    def test_fake_source_conforms_to_protocol(self) -> None:
        source: SplitStatsDataSource = FakeSplitSource()
        assert hasattr(source, "batting_stats_vs_lhp")
        assert hasattr(source, "batting_stats_vs_rhp")

    def test_returns_correct_data(self) -> None:
        lhp_stats = [_make_batting(player_id="p1", pa=200)]
        rhp_stats = [_make_batting(player_id="p1", pa=400)]
        source = FakeSplitSource(
            vs_lhp={2024: lhp_stats},
            vs_rhp={2024: rhp_stats},
        )
        assert source.batting_stats_vs_lhp(2024) == lhp_stats
        assert source.batting_stats_vs_rhp(2024) == rhp_stats

    def test_missing_year_returns_empty(self) -> None:
        source = FakeSplitSource()
        assert source.batting_stats_vs_lhp(2024) == []
        assert source.batting_stats_vs_rhp(2024) == []


class TestCachedSplitDataSource:
    def test_delegates_on_cache_miss(self) -> None:
        lhp_stats = [_make_batting(player_id="p1")]
        rhp_stats = [_make_batting(player_id="p2")]
        delegate = FakeSplitSource(
            vs_lhp={2024: lhp_stats},
            vs_rhp={2024: rhp_stats},
        )
        cache = FakeCacheStore()
        cached = CachedSplitDataSource(delegate=delegate, cache=cache)

        result_lhp = cached.batting_stats_vs_lhp(2024)
        result_rhp = cached.batting_stats_vs_rhp(2024)

        assert len(result_lhp) == 1
        assert result_lhp[0].player_id == "p1"
        assert len(result_rhp) == 1
        assert result_rhp[0].player_id == "p2"

    def test_serves_from_cache_on_hit(self) -> None:
        lhp_stats = [_make_batting(player_id="p1")]
        delegate = FakeSplitSource(vs_lhp={2024: lhp_stats})
        cache = FakeCacheStore()
        cached = CachedSplitDataSource(delegate=delegate, cache=cache)

        # First call populates cache
        cached.batting_stats_vs_lhp(2024)

        # Replace delegate with empty source
        cached._delegate = FakeSplitSource()

        # Should still return cached data
        result = cached.batting_stats_vs_lhp(2024)
        assert len(result) == 1
        assert result[0].player_id == "p1"

    def test_round_trips_all_fields(self) -> None:
        original = _make_batting(player_id="p1", name="Test", year=2024, pa=300)
        delegate = FakeSplitSource(vs_lhp={2024: [original]})
        cache = FakeCacheStore()
        cached = CachedSplitDataSource(delegate=delegate, cache=cache)

        # Populate cache
        cached.batting_stats_vs_lhp(2024)
        # Force cache hit
        cached._delegate = FakeSplitSource()
        result = cached.batting_stats_vs_lhp(2024)

        assert result[0] == original
