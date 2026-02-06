"""Tests for ADP caching via the cached() wrapper with TupleFieldDataclassListSerializer."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, overload

from fantasy_baseball_manager.adp.models import ADPEntry
from fantasy_baseball_manager.cache.serialization import TupleFieldDataclassListSerializer
from fantasy_baseball_manager.cache.wrapper import cached
from fantasy_baseball_manager.context import init_context, reset_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.result import Err, Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.player.identity import Player


class FakeADPDataSource:
    """Fake DataSource[ADPEntry] for testing."""

    def __init__(self, entries: list[ADPEntry]) -> None:
        self._entries = entries
        self.call_count = 0

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[ADPEntry]] | Err[DataSourceError]: ...
    @overload
    def __call__(self, query: list[Player]) -> Ok[list[ADPEntry]] | Err[DataSourceError]: ...
    @overload
    def __call__(self, query: Player) -> Ok[ADPEntry] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[ADPEntry]] | Ok[ADPEntry] | Err[DataSourceError]:
        self.call_count += 1
        return Ok(self._entries)


def _sample_entries() -> list[ADPEntry]:
    return [
        ADPEntry(name="Mike Trout", adp=1.5, positions=("OF",), percent_drafted=99.8),
        ADPEntry(name="Shohei Ohtani", adp=2.3, positions=("DH", "SP"), percent_drafted=99.5),
    ]


def _make_cached_source(
    delegate: FakeADPDataSource,
) -> ...:
    serializer = TupleFieldDataclassListSerializer(ADPEntry, tuple_fields=("positions",))
    return cached(delegate, namespace="adp_test", ttl_seconds=86400, serializer=serializer)


class TestCachedADPDataSource:
    def setup_method(self) -> None:
        fd, tmp_name = tempfile.mkstemp(suffix=".db")
        import os

        os.close(fd)
        self._tmp_path = Path(tmp_name)
        init_context(year=2025, db_path=self._tmp_path)

    def teardown_method(self) -> None:
        reset_context()
        self._tmp_path.unlink(missing_ok=True)

    def test_cache_miss_delegates(self) -> None:
        entries = _sample_entries()
        delegate = FakeADPDataSource(entries)
        source = _make_cached_source(delegate)

        result = source(ALL_PLAYERS)

        assert result.is_ok()
        assert result.unwrap() == entries
        assert delegate.call_count == 1

    def test_cache_hit_returns_without_delegating(self) -> None:
        entries = _sample_entries()
        delegate = FakeADPDataSource(entries)
        source = _make_cached_source(delegate)

        source(ALL_PLAYERS)  # populate cache
        result = source(ALL_PLAYERS)  # should hit cache

        assert result.is_ok()
        assert result.unwrap() == entries
        assert delegate.call_count == 1

    def test_round_trip_preserves_positions_as_tuples(self) -> None:
        entries = _sample_entries()
        delegate = FakeADPDataSource(entries)
        source = _make_cached_source(delegate)

        source(ALL_PLAYERS)  # populate cache
        result = source(ALL_PLAYERS)  # read from cache

        restored = result.unwrap()
        assert isinstance(restored[0].positions, tuple)
        assert restored[0].positions == ("OF",)
        assert isinstance(restored[1].positions, tuple)
        assert restored[1].positions == ("DH", "SP")

    def test_preserves_percent_drafted(self) -> None:
        entries = _sample_entries()
        delegate = FakeADPDataSource(entries)
        source = _make_cached_source(delegate)

        source(ALL_PLAYERS)
        result = source(ALL_PLAYERS)

        restored = result.unwrap()
        assert restored[0].percent_drafted == 99.8
        assert restored[1].percent_drafted == 99.5

    def test_handles_none_percent_drafted(self) -> None:
        entries = [ADPEntry(name="Player", adp=10.0, positions=("1B",), percent_drafted=None)]
        delegate = FakeADPDataSource(entries)
        source = _make_cached_source(delegate)

        source(ALL_PLAYERS)
        result = source(ALL_PLAYERS)

        restored = result.unwrap()
        assert restored[0].percent_drafted is None
