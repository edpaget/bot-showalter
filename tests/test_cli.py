from typer.testing import CliRunner

from fantasy_baseball_manager.cli import app

runner = CliRunner()


class TestRootCli:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Fantasy baseball manager" in result.output

    def test_players_help(self) -> None:
        result = runner.invoke(app, ["players", "--help"])
        assert result.exit_code == 0
        assert "Player projection and valuation commands" in result.output

    def test_players_project_help(self) -> None:
        result = runner.invoke(app, ["players", "project", "--help"])
        assert result.exit_code == 0
        assert "projections" in result.output.lower()

    def test_teams_help(self) -> None:
        result = runner.invoke(app, ["teams", "--help"])
        assert result.exit_code == 0
        assert "Team roster and comparison commands" in result.output
