"""Tests for cached projection source."""

from datetime import UTC, datetime

import pytest

from fantasy_baseball_manager.cache.sources import (
    CachedProjectionSource,
    _deserialize_projection_data,
    _serialize_projection_data,
)
from fantasy_baseball_manager.projections.models import (
    BattingProjection,
    PitchingProjection,
    ProjectionData,
    ProjectionSystem,
)


@pytest.fixture
def sample_projection_data() -> ProjectionData:
    """Create sample projection data for testing."""
    return ProjectionData(
        batting=(
            BattingProjection(
                player_id="15640",
                mlbam_id="592450",
                name="Aaron Judge",
                team="NYY",
                position="OF",
                g=141,
                pa=635,
                ab=510,
                h=145,
                singles=78,
                doubles=24,
                triples=1,
                hr=43,
                r=110,
                rbi=104,
                sb=9,
                cs=2,
                bb=112,
                so=156,
                hbp=6,
                sf=4,
                sh=2,
                obp=0.417,
                slg=0.587,
                ops=1.004,
                woba=0.415,
                war=6.7,
            ),
        ),
        pitching=(
            PitchingProjection(
                player_id="22267",
                mlbam_id="669373",
                name="Tarik Skubal",
                team="DET",
                g=32,
                gs=32,
                ip=199.8,
                w=14,
                l=9,
                sv=0,
                hld=0,
                so=243,
                bb=44,
                hbp=7,
                h=160,
                er=62,
                hr=20,
                era=2.80,
                whip=1.02,
                fip=2.78,
                war=5.9,
            ),
        ),
        system=ProjectionSystem.STEAMER,
        fetched_at=datetime(2025, 3, 1, 12, 0, 0, tzinfo=UTC),
    )


class TestProjectionSerialization:
    """Tests for projection data serialization."""

    def test_round_trip_preserves_data(self, sample_projection_data: ProjectionData) -> None:
        """Serializing and deserializing preserves all data."""
        serialized = _serialize_projection_data(sample_projection_data)
        deserialized = _deserialize_projection_data(serialized)

        assert deserialized.system == sample_projection_data.system
        assert deserialized.fetched_at == sample_projection_data.fetched_at
        assert len(deserialized.batting) == len(sample_projection_data.batting)
        assert len(deserialized.pitching) == len(sample_projection_data.pitching)

        # Check batting
        orig_bat = sample_projection_data.batting[0]
        deser_bat = deserialized.batting[0]
        assert deser_bat.player_id == orig_bat.player_id
        assert deser_bat.mlbam_id == orig_bat.mlbam_id
        assert deser_bat.name == orig_bat.name
        assert deser_bat.hr == orig_bat.hr
        assert deser_bat.obp == pytest.approx(orig_bat.obp)

        # Check pitching
        orig_pit = sample_projection_data.pitching[0]
        deser_pit = deserialized.pitching[0]
        assert deser_pit.player_id == orig_pit.player_id
        assert deser_pit.name == orig_pit.name
        assert deser_pit.era == pytest.approx(orig_pit.era)
        assert deser_pit.sv == orig_pit.sv

    def test_handles_none_mlbam_id(self) -> None:
        """Serialization handles None MLBAM ID."""
        data = ProjectionData(
            batting=(
                BattingProjection(
                    player_id="12345",
                    mlbam_id=None,
                    name="Test Player",
                    team="TST",
                    position="1B",
                    g=100,
                    pa=400,
                    ab=350,
                    h=90,
                    singles=50,
                    doubles=20,
                    triples=2,
                    hr=20,
                    r=50,
                    rbi=60,
                    sb=5,
                    cs=2,
                    bb=40,
                    so=80,
                    hbp=5,
                    sf=3,
                    sh=1,
                    obp=0.350,
                    slg=0.450,
                    ops=0.800,
                    woba=0.340,
                    war=2.5,
                ),
            ),
            pitching=(),
            system=ProjectionSystem.ZIPS,
            fetched_at=datetime(2025, 3, 1, tzinfo=UTC),
        )

        serialized = _serialize_projection_data(data)
        deserialized = _deserialize_projection_data(serialized)

        assert deserialized.batting[0].mlbam_id is None


class TestCachedProjectionSource:
    """Tests for CachedProjectionSource."""

    def test_returns_cached_data_on_hit(self, sample_projection_data: ProjectionData) -> None:
        """Returns cached data without calling delegate on cache hit."""

        class FakeCache:
            def __init__(self, cached_data: str | None):
                self._data = cached_data
                self.put_called = False

            def get(self, namespace: str, key: str) -> str | None:
                return self._data

            def put(self, namespace: str, key: str, value: str, ttl: int) -> None:
                self.put_called = True
                self._data = value

        class FakeDelegate:
            def __init__(self) -> None:
                self.fetch_called = False

            def fetch_projections(self) -> ProjectionData:
                self.fetch_called = True
                return sample_projection_data

        cached_data = _serialize_projection_data(sample_projection_data)
        cache = FakeCache(cached_data)
        delegate = FakeDelegate()

        source = CachedProjectionSource(
            delegate=delegate,
            cache=cache,  # type: ignore[arg-type]
            cache_key="steamer",
        )
        result = source.fetch_projections()

        assert not delegate.fetch_called
        assert result.system == ProjectionSystem.STEAMER
        assert len(result.batting) == 1

    def test_fetches_and_caches_on_miss(self, sample_projection_data: ProjectionData) -> None:
        """Fetches from delegate and caches on cache miss."""

        class FakeCache:
            def __init__(self) -> None:
                self._data: str | None = None
                self.put_called = False

            def get(self, namespace: str, key: str) -> str | None:
                return self._data

            def put(self, namespace: str, key: str, value: str, ttl: int) -> None:
                self.put_called = True
                self._data = value

        class FakeDelegate:
            def __init__(self, data: ProjectionData) -> None:
                self._data = data
                self.fetch_called = False

            def fetch_projections(self) -> ProjectionData:
                self.fetch_called = True
                return self._data

        cache = FakeCache()
        delegate = FakeDelegate(sample_projection_data)

        source = CachedProjectionSource(
            delegate=delegate,
            cache=cache,  # type: ignore[arg-type]
            cache_key="steamer",
        )
        result = source.fetch_projections()

        assert delegate.fetch_called
        assert cache.put_called
        assert result.system == ProjectionSystem.STEAMER
