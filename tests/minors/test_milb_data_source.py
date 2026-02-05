"""Tests for new-style minor league DataSource classes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fantasy_baseball_manager.cache.wrapper import cached
from fantasy_baseball_manager.context import Context, new_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.minors.cached_data_source import (
    _deserialize_batting,
    _serialize_batting,
)
from fantasy_baseball_manager.minors.data_source import (
    MinorLeagueBattingDataSource,
    MinorLeaguePitchingDataSource,
    create_milb_batting_source,
    create_milb_pitching_source,
)
from fantasy_baseball_manager.minors.types import (
    MinorLeagueBatterSeasonStats,
    MinorLeagueLevel,
    MinorLeaguePitcherSeasonStats,
)
from fantasy_baseball_manager.result import Err, Ok


def _sample_batting() -> list[MinorLeagueBatterSeasonStats]:
    return [
        MinorLeagueBatterSeasonStats(
            player_id="12345",
            name="Test Player",
            season=2024,
            age=23,
            level=MinorLeagueLevel.AAA,
            team="Syracuse Mets",
            league="International League",
            pa=500,
            ab=450,
            h=130,
            singles=90,
            doubles=25,
            triples=5,
            hr=10,
            rbi=60,
            r=70,
            bb=40,
            so=100,
            hbp=5,
            sf=5,
            sb=15,
            cs=5,
            avg=0.289,
            obp=0.360,
            slg=0.440,
        ),
    ]


def _sample_pitching() -> list[MinorLeaguePitcherSeasonStats]:
    return [
        MinorLeaguePitcherSeasonStats(
            player_id="54321",
            name="Test Pitcher",
            season=2024,
            age=24,
            level=MinorLeagueLevel.AA,
            team="Binghamton Rumble Ponies",
            league="Eastern League",
            g=25,
            gs=25,
            ip=140.333,
            w=10,
            losses=5,
            sv=0,
            h=120,
            r=55,
            er=50,
            hr=12,
            bb=40,
            so=150,
            hbp=5,
            era=3.21,
            whip=1.14,
        ),
    ]


class TestMinorLeagueBattingDataSource:
    """Tests for MinorLeagueBattingDataSource class."""

    def test_is_callable(self, test_context: Context) -> None:
        """MinorLeagueBattingDataSource is callable."""
        source = MinorLeagueBattingDataSource()
        assert callable(source)

    def test_factory_returns_data_source(self, test_context: Context) -> None:
        """create_milb_batting_source() returns a MinorLeagueBattingDataSource."""
        source = create_milb_batting_source()
        assert isinstance(source, MinorLeagueBattingDataSource)

    def test_all_players_returns_sequence(self, test_context: Context) -> None:
        """ALL_PLAYERS query returns Ok with sequence of MinorLeagueBatterSeasonStats."""
        with patch.object(
            MinorLeagueBattingDataSource,
            "__init__",
            lambda self, timeout=30: setattr(self, "_delegate", MagicMock()),
        ):
            source = MinorLeagueBattingDataSource()
            source._delegate.batting_stats_all_levels.return_value = _sample_batting()  # type: ignore[union-attr]

            result = source(ALL_PLAYERS)

            assert result.is_ok()
            stats = result.unwrap()
            assert len(stats) == 1
            assert isinstance(stats[0], MinorLeagueBatterSeasonStats)
            assert stats[0].player_id == "12345"
            assert stats[0].name == "Test Player"
            assert stats[0].hr == 10

    def test_uses_year_from_context(self, test_context: Context) -> None:
        """Source uses year from ambient context."""
        with patch.object(
            MinorLeagueBattingDataSource,
            "__init__",
            lambda self, timeout=30: setattr(self, "_delegate", MagicMock()),
        ):
            source = MinorLeagueBattingDataSource()
            source._delegate.batting_stats_all_levels.return_value = []  # type: ignore[union-attr]

            # Default context year is 2025
            source(ALL_PLAYERS)
            source._delegate.batting_stats_all_levels.assert_called_with(2025)  # type: ignore[union-attr]

            # Switch context to different year
            with new_context(year=2023):
                source(ALL_PLAYERS)
                source._delegate.batting_stats_all_levels.assert_called_with(2023)  # type: ignore[union-attr]

    def test_returns_error_on_failure(self, test_context: Context) -> None:
        """Returns Err on delegate failure."""
        with patch.object(
            MinorLeagueBattingDataSource,
            "__init__",
            lambda self, timeout=30: setattr(self, "_delegate", MagicMock()),
        ):
            source = MinorLeagueBattingDataSource()
            source._delegate.batting_stats_all_levels.side_effect = Exception("Network error")  # type: ignore[union-attr]

            result = source(ALL_PLAYERS)

            assert result.is_err()
            error = result.unwrap_err()
            assert isinstance(error, DataSourceError)
            assert "Network error" in str(error)

    def test_single_player_query_returns_error(self, test_context: Context) -> None:
        """Single player queries return Err (batch-only source)."""
        with patch.object(
            MinorLeagueBattingDataSource,
            "__init__",
            lambda self, timeout=30: setattr(self, "_delegate", MagicMock()),
        ):
            source = MinorLeagueBattingDataSource()
            mock_player = MagicMock()

            result = source(mock_player)

            assert result.is_err()
            error = result.unwrap_err()
            assert "Only ALL_PLAYERS" in str(error)


class TestMinorLeaguePitchingDataSource:
    """Tests for MinorLeaguePitchingDataSource class."""

    def test_is_callable(self, test_context: Context) -> None:
        """MinorLeaguePitchingDataSource is callable."""
        source = MinorLeaguePitchingDataSource()
        assert callable(source)

    def test_factory_returns_data_source(self, test_context: Context) -> None:
        """create_milb_pitching_source() returns a MinorLeaguePitchingDataSource."""
        source = create_milb_pitching_source()
        assert isinstance(source, MinorLeaguePitchingDataSource)

    def test_all_players_returns_sequence(self, test_context: Context) -> None:
        """ALL_PLAYERS query returns Ok with sequence of MinorLeaguePitcherSeasonStats."""
        with patch.object(
            MinorLeaguePitchingDataSource,
            "__init__",
            lambda self, timeout=30: setattr(self, "_delegate", MagicMock()),
        ):
            source = MinorLeaguePitchingDataSource()
            source._delegate.pitching_stats_all_levels.return_value = _sample_pitching()  # type: ignore[union-attr]

            result = source(ALL_PLAYERS)

            assert result.is_ok()
            stats = result.unwrap()
            assert len(stats) == 1
            assert isinstance(stats[0], MinorLeaguePitcherSeasonStats)
            assert stats[0].player_id == "54321"
            assert stats[0].name == "Test Pitcher"
            assert stats[0].so == 150

    def test_uses_year_from_context(self, test_context: Context) -> None:
        """Source uses year from ambient context."""
        with patch.object(
            MinorLeaguePitchingDataSource,
            "__init__",
            lambda self, timeout=30: setattr(self, "_delegate", MagicMock()),
        ):
            source = MinorLeaguePitchingDataSource()
            source._delegate.pitching_stats_all_levels.return_value = []  # type: ignore[union-attr]

            # Default context year is 2025
            source(ALL_PLAYERS)
            source._delegate.pitching_stats_all_levels.assert_called_with(2025)  # type: ignore[union-attr]

            # Switch context to different year
            with new_context(year=2023):
                source(ALL_PLAYERS)
                source._delegate.pitching_stats_all_levels.assert_called_with(2023)  # type: ignore[union-attr]

    def test_returns_error_on_failure(self, test_context: Context) -> None:
        """Returns Err on delegate failure."""
        with patch.object(
            MinorLeaguePitchingDataSource,
            "__init__",
            lambda self, timeout=30: setattr(self, "_delegate", MagicMock()),
        ):
            source = MinorLeaguePitchingDataSource()
            source._delegate.pitching_stats_all_levels.side_effect = Exception("Network error")  # type: ignore[union-attr]

            result = source(ALL_PLAYERS)

            assert result.is_err()
            error = result.unwrap_err()
            assert isinstance(error, DataSourceError)
            assert "Network error" in str(error)


class _MiLBBatterSerializer:
    """Serializer for MinorLeagueBatterSeasonStats using existing functions."""

    def serialize(self, value: list[MinorLeagueBatterSeasonStats]) -> str:
        return _serialize_batting(value)

    def deserialize(self, data: str) -> list[MinorLeagueBatterSeasonStats]:
        return _deserialize_batting(data)


class TestCachedMinorLeagueBattingSource:
    """Tests for cached minor league batting DataSource."""

    def test_caching_integrates_with_batting_source(self, test_context: Context) -> None:
        """cached() wrapper works with MinorLeagueBattingDataSource."""
        call_count = 0

        class MockMiLBBattingSource:
            """Mock DataSource with proper overloads."""

            def __call__(
                self, query: type[ALL_PLAYERS]
            ) -> Ok[list[MinorLeagueBatterSeasonStats]] | Err[DataSourceError]:
                nonlocal call_count
                call_count += 1
                return Ok(_sample_batting())

        mock_source = MockMiLBBattingSource()
        serializer = _MiLBBatterSerializer()
        cached_source = cached(
            mock_source,  # type: ignore[arg-type]
            namespace="milb_batting",
            ttl_seconds=3600,
            serializer=serializer,
        )

        # First call fetches from source
        result1 = cached_source(ALL_PLAYERS)
        assert result1.is_ok()
        assert call_count == 1

        # Second call hits cache
        result2 = cached_source(ALL_PLAYERS)
        assert result2.is_ok()
        assert call_count == 1  # No additional call

        # Data matches
        assert result1.unwrap() == result2.unwrap()

    def test_cache_key_scoped_by_year(self, test_context: Context) -> None:
        """Cache entries are scoped by year from context."""
        call_count = 0

        class MockMiLBBattingSource:
            def __call__(
                self, query: type[ALL_PLAYERS]
            ) -> Ok[list[MinorLeagueBatterSeasonStats]] | Err[DataSourceError]:
                nonlocal call_count
                call_count += 1
                return Ok([])

        mock_source = MockMiLBBattingSource()
        serializer = _MiLBBatterSerializer()
        cached_source = cached(
            mock_source,  # type: ignore[arg-type]
            namespace="milb_batting",
            ttl_seconds=3600,
            serializer=serializer,
        )

        # Fetch 2025 (default context year)
        cached_source(ALL_PLAYERS)
        assert call_count == 1

        # Fetch 2024 - different cache key
        with new_context(year=2024):
            cached_source(ALL_PLAYERS)
        assert call_count == 2

        # Fetch 2025 again - cache hit
        cached_source(ALL_PLAYERS)
        assert call_count == 2
