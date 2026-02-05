"""Tests for the cache wrapper module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from fantasy_baseball_manager.cache.serialization import DataclassListSerializer
from fantasy_baseball_manager.cache.wrapper import cached
from fantasy_baseball_manager.context import Context, get_context, init_context, new_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class MockStats:
    """Mock stats dataclass for testing."""

    player_id: str
    value: int


class TestCached:
    """Tests for the cached() wrapper function."""

    @pytest.fixture
    def mock_source(self) -> ...:
        """Create a mock DataSource that tracks calls."""
        calls: list[object] = []

        def source(query: ...) -> Ok[list[MockStats]] | Err[DataSourceError]:
            calls.append(query)
            if query is ALL_PLAYERS:
                return Ok([MockStats("p1", 100), MockStats("p2", 200)])
            return Err(DataSourceError("Unsupported query"))

        source.calls = calls  # type: ignore[attr-defined]
        return source

    @pytest.fixture
    def serializer(self) -> DataclassListSerializer[MockStats]:
        """Create serializer for MockStats."""
        return DataclassListSerializer(MockStats)

    def test_caches_all_players_query(
        self,
        test_context: Context,
        mock_source: ...,
        serializer: DataclassListSerializer[MockStats],
    ) -> None:
        """ALL_PLAYERS queries are cached."""
        wrapped = cached(mock_source, "test", ttl_seconds=3600, serializer=serializer)

        # First call - fetches from source
        result1 = wrapped(ALL_PLAYERS)
        assert result1.is_ok()
        assert len(mock_source.calls) == 1

        # Second call - should hit cache
        result2 = wrapped(ALL_PLAYERS)
        assert result2.is_ok()
        assert len(mock_source.calls) == 1  # No additional call

        # Results should match
        assert result1.unwrap() == result2.unwrap()

    def test_respects_cache_disabled(
        self,
        tmp_path: Path,
        mock_source: ...,
        serializer: DataclassListSerializer[MockStats],
    ) -> None:
        """Cache is bypassed when cache_enabled=False."""
        init_context(year=2025, cache_enabled=False, db_path=tmp_path / "cache.db")
        wrapped = cached(mock_source, "test", ttl_seconds=3600, serializer=serializer)

        # Both calls go to source
        wrapped(ALL_PLAYERS)
        wrapped(ALL_PLAYERS)

        assert len(mock_source.calls) == 2

    def test_respects_cache_invalidated(
        self,
        tmp_path: Path,
        mock_source: ...,
        serializer: DataclassListSerializer[MockStats],
    ) -> None:
        """Cache reads are skipped when cache_invalidated=True (refresh mode)."""
        # First, populate cache
        init_context(year=2025, cache_enabled=True, db_path=tmp_path / "cache.db")
        wrapped = cached(mock_source, "test", ttl_seconds=3600, serializer=serializer)
        wrapped(ALL_PLAYERS)
        assert len(mock_source.calls) == 1

        # Now enable invalidation mode
        init_context(
            year=2025,
            cache_enabled=True,
            cache_invalidated=True,
            db_path=tmp_path / "cache.db",
        )

        # Should fetch fresh despite cache existing
        wrapped(ALL_PLAYERS)
        assert len(mock_source.calls) == 2

    def test_cache_key_includes_year(
        self,
        tmp_path: Path,
        mock_source: ...,
        serializer: DataclassListSerializer[MockStats],
    ) -> None:
        """Cache keys are scoped by year."""
        init_context(year=2025, cache_enabled=True, db_path=tmp_path / "cache.db")
        wrapped = cached(mock_source, "test", ttl_seconds=3600, serializer=serializer)

        # Fetch 2025 data
        wrapped(ALL_PLAYERS)
        assert len(mock_source.calls) == 1

        # Switch to 2024 - should not hit 2025 cache
        with new_context(year=2024):
            wrapped(ALL_PLAYERS)

        assert len(mock_source.calls) == 2

    def test_passes_through_non_all_players_queries(
        self,
        test_context: Context,
        mock_source: ...,
        serializer: DataclassListSerializer[MockStats],
    ) -> None:
        """Non-ALL_PLAYERS queries are not cached."""
        from fantasy_baseball_manager.player.identity import Player

        wrapped = cached(mock_source, "test", ttl_seconds=3600, serializer=serializer)
        player = Player(name="Test", yahoo_id="123")

        # Single player query
        result = wrapped(player)

        # Should pass through to source
        assert len(mock_source.calls) == 1
        assert mock_source.calls[0] is player

    def test_handles_source_errors(
        self,
        test_context: Context,
        serializer: DataclassListSerializer[MockStats],
    ) -> None:
        """Errors from source are propagated, not cached."""

        def failing_source(query: ...) -> Err[DataSourceError]:
            return Err(DataSourceError("Network error"))

        wrapped = cached(failing_source, "test", ttl_seconds=3600, serializer=serializer)

        result = wrapped(ALL_PLAYERS)
        assert result.is_err()
        assert "Network error" in str(result.unwrap_err())


class TestCacheWithMultipleNamespaces:
    """Tests for caching with different namespaces."""

    def test_namespaces_are_isolated(self, test_context: Context) -> None:
        """Different namespaces don't share cache entries."""
        serializer = DataclassListSerializer(MockStats)

        batting_calls: list[object] = []
        pitching_calls: list[object] = []

        def batting_source(query: ...) -> Ok[list[MockStats]]:
            batting_calls.append(query)
            return Ok([MockStats("batter", 100)])

        def pitching_source(query: ...) -> Ok[list[MockStats]]:
            pitching_calls.append(query)
            return Ok([MockStats("pitcher", 50)])

        batting = cached(batting_source, "batting", ttl_seconds=3600, serializer=serializer)  # type: ignore[arg-type]
        pitching = cached(pitching_source, "pitching", ttl_seconds=3600, serializer=serializer)  # type: ignore[arg-type]

        # Fetch batting - caches
        batting(ALL_PLAYERS)
        batting(ALL_PLAYERS)  # Should hit cache

        # Fetch pitching - separate cache
        pitching(ALL_PLAYERS)
        pitching(ALL_PLAYERS)  # Should hit cache

        assert len(batting_calls) == 1
        assert len(pitching_calls) == 1
