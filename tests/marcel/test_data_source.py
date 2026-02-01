import pytest

from fantasy_baseball_manager.marcel.data_source import PybaseballDataSource


@pytest.mark.integration
class TestPybaseballDataSource:
    """Integration tests that hit the real pybaseball API.

    Run with: uv run pytest -m integration
    """

    def test_batting_stats_returns_data(self) -> None:
        ds = PybaseballDataSource()
        stats = ds.batting_stats(2023)
        assert len(stats) > 100
        # Check a known player exists
        names = {s.name for s in stats}
        assert "Shohei Ohtani" in names or "Ronald Acuna Jr." in names

    def test_pitching_stats_returns_data(self) -> None:
        ds = PybaseballDataSource()
        stats = ds.pitching_stats(2023)
        assert len(stats) > 100

    def test_team_batting_returns_30_teams(self) -> None:
        ds = PybaseballDataSource()
        stats = ds.team_batting(2023)
        assert len(stats) == 30

    def test_team_pitching_returns_30_teams(self) -> None:
        ds = PybaseballDataSource()
        stats = ds.team_pitching(2023)
        assert len(stats) == 30
