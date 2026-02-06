import pytest

from fantasy_baseball_manager.context import new_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS
from fantasy_baseball_manager.marcel.data_source import (
    BattingDataSource,
    PitchingDataSource,
    TeamBattingDataSource,
    TeamPitchingDataSource,
)


@pytest.mark.integration
class TestBattingDataSource:
    """Integration tests that hit the real pybaseball API.

    Run with: uv run pytest -m integration
    """

    def test_batting_stats_returns_data(self) -> None:
        ds = BattingDataSource()
        with new_context(year=2023):
            result = ds(ALL_PLAYERS)
        assert result.is_ok()
        stats = result.unwrap()
        assert len(stats) > 100
        names = {s.name for s in stats}
        assert "Shohei Ohtani" in names or "Ronald Acuna Jr." in names


@pytest.mark.integration
class TestPitchingDataSource:
    def test_pitching_stats_returns_data(self) -> None:
        ds = PitchingDataSource()
        with new_context(year=2023):
            result = ds(ALL_PLAYERS)
        assert result.is_ok()
        stats = result.unwrap()
        assert len(stats) > 100


@pytest.mark.integration
class TestTeamBattingDataSource:
    def test_team_batting_returns_30_teams(self) -> None:
        ds = TeamBattingDataSource()
        with new_context(year=2023):
            result = ds(ALL_PLAYERS)
        assert result.is_ok()
        stats = result.unwrap()
        assert len(stats) == 30


@pytest.mark.integration
class TestTeamPitchingDataSource:
    def test_team_pitching_returns_30_teams(self) -> None:
        ds = TeamPitchingDataSource()
        with new_context(year=2023):
            result = ds(ALL_PLAYERS)
        assert result.is_ok()
        stats = result.unwrap()
        assert len(stats) == 30
