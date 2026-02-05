"""Tests for new-style batting DataSource."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

from fantasy_baseball_manager.cache.serialization import DataclassListSerializer
from fantasy_baseball_manager.cache.wrapper import cached
from fantasy_baseball_manager.context import Context, new_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.marcel.data_source import (
    BattingSeasonStats,
    create_batting_source,
)
from fantasy_baseball_manager.result import Ok

if TYPE_CHECKING:
    from collections.abc import Sequence


class TestCreateBattingSource:
    """Tests for the create_batting_source factory."""

    def test_returns_callable_data_source(self, test_context: Context) -> None:
        """Factory returns a callable DataSource."""
        source = create_batting_source()
        assert callable(source)

    def test_all_players_returns_sequence(self, test_context: Context) -> None:
        """ALL_PLAYERS query returns Ok with sequence of BattingSeasonStats."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            # Setup mock DataFrame
            mock_df = MagicMock()
            mock_df.iterrows.return_value = iter(
                [
                    (
                        0,
                        {
                            "IDfg": "12345",
                            "Name": "Test Player",
                            "Age": 28,
                            "PA": 600,
                            "AB": 540,
                            "H": 160,
                            "2B": 30,
                            "3B": 5,
                            "HR": 25,
                            "BB": 50,
                            "SO": 120,
                            "HBP": 5,
                            "SF": 3,
                            "SH": 2,
                            "SB": 10,
                            "CS": 3,
                            "R": 80,
                            "RBI": 90,
                            "Team": "NYY",
                        },
                    )
                ]
            )
            mock_pb.batting_stats.return_value = mock_df

            source = create_batting_source()
            result = source(ALL_PLAYERS)

            assert result.is_ok()
            stats = cast("Sequence[BattingSeasonStats]", result.unwrap())
            assert len(stats) == 1
            assert isinstance(stats[0], BattingSeasonStats)
            assert stats[0].player_id == "12345"
            assert stats[0].name == "Test Player"
            assert stats[0].hr == 25

    def test_uses_year_from_context(self, test_context: Context) -> None:
        """Source uses year from ambient context."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_df = MagicMock()
            mock_df.iterrows.return_value = iter([])
            mock_pb.batting_stats.return_value = mock_df

            source = create_batting_source()

            # Default context year is 2025
            source(ALL_PLAYERS)
            mock_pb.batting_stats.assert_called_with(2025, qual=0)

            # Switch context to different year
            with new_context(year=2023):
                source(ALL_PLAYERS)
                mock_pb.batting_stats.assert_called_with(2023, qual=0)

    def test_returns_error_on_failure(self, test_context: Context) -> None:
        """Returns Err on pybaseball failure."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_pb.batting_stats.side_effect = Exception("Network error")

            source = create_batting_source()
            result = source(ALL_PLAYERS)

            assert result.is_err()
            error = result.unwrap_err()
            assert isinstance(error, DataSourceError)
            assert "Network error" in str(error)

    def test_single_player_query_not_supported(self, test_context: Context) -> None:
        """Single player queries return Err (batch-only source)."""
        from fantasy_baseball_manager.player.identity import Player

        source = create_batting_source()
        player = Player(name="Test", yahoo_id="123", fangraphs_id="12345")

        result = source(player)

        assert result.is_err()
        assert "not supported" in str(result.unwrap_err()).lower()

    def test_calculates_singles_from_hits(self, test_context: Context) -> None:
        """Singles are calculated as H - 2B - 3B - HR."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_df = MagicMock()
            mock_df.iterrows.return_value = iter(
                [
                    (
                        0,
                        {
                            "IDfg": "1",
                            "Name": "Test",
                            "Age": 25,
                            "PA": 100,
                            "AB": 90,
                            "H": 30,  # Total hits
                            "2B": 8,
                            "3B": 2,
                            "HR": 5,  # 8+2+5=15 extra base hits
                            "BB": 5,
                            "SO": 20,
                            "HBP": 2,
                            "SF": 1,
                            "SH": 0,
                            "SB": 3,
                            "CS": 1,
                            "R": 15,
                            "RBI": 20,
                            "Team": "BOS",
                        },
                    )
                ]
            )
            mock_pb.batting_stats.return_value = mock_df

            source = create_batting_source()
            result = source(ALL_PLAYERS)

            stats = cast("Sequence[BattingSeasonStats]", result.unwrap())
            assert stats[0].singles == 15  # 30 - 8 - 2 - 5


class TestCachedBattingSource:
    """Tests for cached batting DataSource."""

    def test_caching_integrates_with_batting_source(self, test_context: Context) -> None:
        """cached() wrapper works with create_batting_source()."""
        call_count = 0

        def mock_source(query: object) -> Ok[list[BattingSeasonStats]]:
            nonlocal call_count
            call_count += 1
            return Ok(
                [
                    BattingSeasonStats(
                        player_id="1",
                        name="Test",
                        year=2025,
                        age=25,
                        pa=100,
                        ab=90,
                        h=25,
                        singles=15,
                        doubles=5,
                        triples=2,
                        hr=3,
                        bb=5,
                        so=20,
                        hbp=2,
                        sf=1,
                        sh=0,
                        sb=3,
                        cs=1,
                        r=15,
                        rbi=20,
                        team="BOS",
                    )
                ]
            )

        serializer = DataclassListSerializer(BattingSeasonStats)
        cached_source = cached(
            mock_source,
            namespace="batting_stats",
            ttl_seconds=3600,
            serializer=serializer,  # type: ignore[arg-type]
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

        def mock_source(query: object) -> Ok[list[BattingSeasonStats]]:
            nonlocal call_count
            call_count += 1
            return Ok([])

        serializer = DataclassListSerializer(BattingSeasonStats)
        cached_source = cached(
            mock_source,
            namespace="batting_stats",
            ttl_seconds=3600,
            serializer=serializer,  # type: ignore[arg-type]
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
