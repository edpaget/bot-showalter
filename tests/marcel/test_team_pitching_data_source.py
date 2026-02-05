"""Tests for new-style team pitching DataSource."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fantasy_baseball_manager.context import Context, new_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSourceError
from fantasy_baseball_manager.marcel.data_source import (
    PitchingSeasonStats,
    TeamPitchingDataSource,
    create_team_pitching_source,
)


class TestTeamPitchingDataSource:
    """Tests for the TeamPitchingDataSource class."""

    def test_is_callable(self, test_context: Context) -> None:
        """TeamPitchingDataSource is callable."""
        source = TeamPitchingDataSource()
        assert callable(source)

    def test_factory_returns_data_source(self, test_context: Context) -> None:
        """create_team_pitching_source() returns a TeamPitchingDataSource."""
        source = create_team_pitching_source()
        assert isinstance(source, TeamPitchingDataSource)

    def test_all_players_returns_sequence(self, test_context: Context) -> None:
        """ALL_PLAYERS query returns Ok with sequence of PitchingSeasonStats."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_df = MagicMock()
            mock_df.iterrows.return_value = iter(
                [
                    (
                        0,
                        {
                            "teamIDfg": "LAD",
                            "Team": "Dodgers",
                            "IP": 1440.0,
                            "G": 162,
                            "GS": 162,
                            "ER": 600,
                            "H": 1200,
                            "BB": 450,
                            "SO": 1400,
                            "HR": 150,
                            "HBP": 50,
                            "W": 95,
                            "SV": 45,
                            "HLD": 60,
                            "BS": 15,
                        },
                    )
                ]
            )
            mock_pb.team_pitching.return_value = mock_df

            source = create_team_pitching_source()
            result = source(ALL_PLAYERS)

            assert result.is_ok()
            stats = result.unwrap()
            assert len(stats) == 1
            assert isinstance(stats[0], PitchingSeasonStats)
            assert stats[0].player_id == "LAD"
            assert stats[0].name == "Dodgers"
            assert stats[0].so == 1400
            assert stats[0].age == 0  # Team stats have age=0

    def test_uses_year_from_context(self, test_context: Context) -> None:
        """Source uses year from ambient context."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_df = MagicMock()
            mock_df.iterrows.return_value = iter([])
            mock_pb.team_pitching.return_value = mock_df

            source = create_team_pitching_source()

            # Default context year is 2025
            source(ALL_PLAYERS)
            mock_pb.team_pitching.assert_called_with(2025)

            # Switch context to different year
            with new_context(year=2023):
                source(ALL_PLAYERS)
                mock_pb.team_pitching.assert_called_with(2023)

    def test_returns_error_on_failure(self, test_context: Context) -> None:
        """Returns Err on pybaseball failure."""
        with patch("fantasy_baseball_manager.marcel.data_source.pybaseball") as mock_pb:
            mock_pb.team_pitching.side_effect = Exception("Network error")

            source = create_team_pitching_source()
            result = source(ALL_PLAYERS)

            assert result.is_err()
            error = result.unwrap_err()
            assert isinstance(error, DataSourceError)
            assert "Network error" in str(error)
