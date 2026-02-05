"""Tests for new-style team batting DataSource."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fantasy_baseball_manager.context import Context, new_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.marcel.data_source import (
    BattingSeasonStats,
    TeamBattingDataSource,
    create_team_batting_source,
)


class TestTeamBattingDataSource:
    """Tests for the TeamBattingDataSource class."""

    def test_is_callable(self, test_context: Context) -> None:
        """TeamBattingDataSource is callable."""
        source = TeamBattingDataSource()
        assert callable(source)

    def test_factory_returns_data_source(self, test_context: Context) -> None:
        """create_team_batting_source() returns a TeamBattingDataSource."""
        source = create_team_batting_source()
        assert isinstance(source, TeamBattingDataSource)

    def test_all_players_returns_sequence(self, test_context: Context) -> None:
        """ALL_PLAYERS query returns Ok with sequence of BattingSeasonStats."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_df = MagicMock()
            mock_df.iterrows.return_value = iter(
                [
                    (
                        0,
                        {
                            "teamIDfg": "NYY",
                            "Team": "Yankees",
                            "PA": 6000,
                            "AB": 5400,
                            "H": 1400,
                            "2B": 280,
                            "3B": 30,
                            "HR": 200,
                            "BB": 500,
                            "SO": 1200,
                            "HBP": 50,
                            "SF": 30,
                            "SH": 20,
                            "SB": 100,
                            "CS": 30,
                            "R": 800,
                            "RBI": 780,
                        },
                    )
                ]
            )
            mock_pb.team_batting.return_value = mock_df

            source = create_team_batting_source()
            result = source(ALL_PLAYERS)

            assert result.is_ok()
            stats = result.unwrap()
            assert len(stats) == 1
            assert isinstance(stats[0], BattingSeasonStats)
            assert stats[0].player_id == "NYY"
            assert stats[0].name == "Yankees"
            assert stats[0].hr == 200
            assert stats[0].age == 0  # Team stats have age=0

    def test_uses_year_from_context(self, test_context: Context) -> None:
        """Source uses year from ambient context."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_df = MagicMock()
            mock_df.iterrows.return_value = iter([])
            mock_pb.team_batting.return_value = mock_df

            source = create_team_batting_source()

            # Default context year is 2025
            source(ALL_PLAYERS)
            mock_pb.team_batting.assert_called_with(2025)

            # Switch context to different year
            with new_context(year=2023):
                source(ALL_PLAYERS)
                mock_pb.team_batting.assert_called_with(2023)

    def test_returns_error_on_failure(self, test_context: Context) -> None:
        """Returns Err on pybaseball failure."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_pb.team_batting.side_effect = Exception("Network error")

            source = create_team_batting_source()
            result = source(ALL_PLAYERS)

            assert result.is_err()
            error = result.unwrap_err()
            assert isinstance(error, DataSourceError)
            assert "Network error" in str(error)

    def test_calculates_singles_from_hits(self, test_context: Context) -> None:
        """Singles are calculated as H - 2B - 3B - HR."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_df = MagicMock()
            mock_df.iterrows.return_value = iter(
                [
                    (
                        0,
                        {
                            "teamIDfg": "BOS",
                            "Team": "Red Sox",
                            "PA": 1000,
                            "AB": 900,
                            "H": 300,  # Total hits
                            "2B": 80,
                            "3B": 20,
                            "HR": 50,  # 80+20+50=150 extra base hits
                            "BB": 50,
                            "SO": 200,
                            "HBP": 20,
                            "SF": 10,
                            "SH": 5,
                            "SB": 30,
                            "CS": 10,
                            "R": 150,
                            "RBI": 200,
                        },
                    )
                ]
            )
            mock_pb.team_batting.return_value = mock_df

            source = create_team_batting_source()
            result = source(ALL_PLAYERS)

            stats = result.unwrap()
            assert stats[0].singles == 150  # 300 - 80 - 20 - 50
