"""Tests for projection caching via the cached() wrapper with DataclassListSerializer."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, overload

import pytest

from fantasy_baseball_manager.cache.serialization import DataclassListSerializer
from fantasy_baseball_manager.cache.wrapper import cached
from fantasy_baseball_manager.context import init_context, reset_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError

if TYPE_CHECKING:
    from fantasy_baseball_manager.player.identity import Player
from fantasy_baseball_manager.projections.models import (
    BattingProjection,
    PitchingProjection,
)
from fantasy_baseball_manager.result import Err, Ok


class FakeBattingDataSource:
    """Fake DataSource[BattingProjection] for testing."""

    def __init__(self, projections: list[BattingProjection]) -> None:
        self._projections = projections
        self.call_count = 0

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[BattingProjection]] | Err[DataSourceError]: ...
    @overload
    def __call__(self, query: list[Player]) -> Ok[list[BattingProjection]] | Err[DataSourceError]: ...
    @overload
    def __call__(self, query: Player) -> Ok[BattingProjection] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[BattingProjection]] | Ok[BattingProjection] | Err[DataSourceError]:
        self.call_count += 1
        return Ok(self._projections)


class FakePitchingDataSource:
    """Fake DataSource[PitchingProjection] for testing."""

    def __init__(self, projections: list[PitchingProjection]) -> None:
        self._projections = projections
        self.call_count = 0

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[PitchingProjection]] | Err[DataSourceError]: ...
    @overload
    def __call__(self, query: list[Player]) -> Ok[list[PitchingProjection]] | Err[DataSourceError]: ...
    @overload
    def __call__(self, query: Player) -> Ok[PitchingProjection] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[PitchingProjection]] | Ok[PitchingProjection] | Err[DataSourceError]:
        self.call_count += 1
        return Ok(self._projections)


@pytest.fixture
def sample_batting() -> list[BattingProjection]:
    return [
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
    ]


@pytest.fixture
def sample_pitching() -> list[PitchingProjection]:
    return [
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
    ]


class TestBattingProjectionSerialization:
    """Tests for batting projection round-trip via DataclassListSerializer."""

    def test_round_trip_preserves_data(self, sample_batting: list[BattingProjection]) -> None:
        serializer = DataclassListSerializer(BattingProjection)
        serialized = serializer.serialize(sample_batting)
        deserialized = serializer.deserialize(serialized)

        assert len(deserialized) == 1
        orig = sample_batting[0]
        deser = deserialized[0]
        assert deser.player_id == orig.player_id
        assert deser.mlbam_id == orig.mlbam_id
        assert deser.name == orig.name
        assert deser.hr == orig.hr
        assert deser.obp == pytest.approx(orig.obp)

    def test_handles_none_mlbam_id(self) -> None:
        data = [
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
            )
        ]
        serializer = DataclassListSerializer(BattingProjection)
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)

        assert deserialized[0].mlbam_id is None


class TestPitchingProjectionSerialization:
    """Tests for pitching projection round-trip via DataclassListSerializer."""

    def test_round_trip_preserves_data(self, sample_pitching: list[PitchingProjection]) -> None:
        serializer = DataclassListSerializer(PitchingProjection)
        serialized = serializer.serialize(sample_pitching)
        deserialized = serializer.deserialize(serialized)

        assert len(deserialized) == 1
        orig = sample_pitching[0]
        deser = deserialized[0]
        assert deser.player_id == orig.player_id
        assert deser.name == orig.name
        assert deser.era == pytest.approx(orig.era)
        assert deser.sv == orig.sv


class TestCachedBattingProjectionDataSource:
    """Tests for batting projections cached via cached() wrapper."""

    def setup_method(self) -> None:
        fd, tmp_name = tempfile.mkstemp(suffix=".db")
        import os

        os.close(fd)
        self._tmp_path = Path(tmp_name)
        init_context(year=2025, db_path=self._tmp_path)

    def teardown_method(self) -> None:
        reset_context()
        self._tmp_path.unlink(missing_ok=True)

    def test_returns_cached_data_on_hit(self, sample_batting: list[BattingProjection]) -> None:
        delegate = FakeBattingDataSource(sample_batting)
        source = cached(
            delegate,
            namespace="proj_batting_test",
            ttl_seconds=604800,
            serializer=DataclassListSerializer(BattingProjection),
        )

        source(ALL_PLAYERS)  # populate cache
        result = source(ALL_PLAYERS)  # should hit cache

        assert result.is_ok()
        assert len(result.unwrap()) == 1
        assert delegate.call_count == 1  # Only called once

    def test_fetches_and_caches_on_miss(self, sample_batting: list[BattingProjection]) -> None:
        delegate = FakeBattingDataSource(sample_batting)
        source = cached(
            delegate,
            namespace="proj_batting_test2",
            ttl_seconds=604800,
            serializer=DataclassListSerializer(BattingProjection),
        )

        result = source(ALL_PLAYERS)

        assert result.is_ok()
        assert delegate.call_count == 1
        assert result.unwrap()[0].name == "Aaron Judge"


class TestCachedPitchingProjectionDataSource:
    """Tests for pitching projections cached via cached() wrapper."""

    def setup_method(self) -> None:
        fd, tmp_name = tempfile.mkstemp(suffix=".db")
        import os

        os.close(fd)
        self._tmp_path = Path(tmp_name)
        init_context(year=2025, db_path=self._tmp_path)

    def teardown_method(self) -> None:
        reset_context()
        self._tmp_path.unlink(missing_ok=True)

    def test_cache_hit(self, sample_pitching: list[PitchingProjection]) -> None:
        delegate = FakePitchingDataSource(sample_pitching)
        source = cached(
            delegate,
            namespace="proj_pitching_test",
            ttl_seconds=604800,
            serializer=DataclassListSerializer(PitchingProjection),
        )

        source(ALL_PLAYERS)
        result = source(ALL_PLAYERS)

        assert result.is_ok()
        assert delegate.call_count == 1
        assert result.unwrap()[0].name == "Tarik Skubal"
