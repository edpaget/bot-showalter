from typer.testing import CliRunner

from fantasy_baseball_manager.cli import app

runner = CliRunner()


class TestRootCli:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Fantasy baseball manager" in result.output

    def test_project_help(self) -> None:
        result = runner.invoke(app, ["project", "--help"])
        assert result.exit_code == 0
        assert "Projection commands" in result.output

    def test_project_marcel_help(self) -> None:
        result = runner.invoke(app, ["project", "marcel", "--help"])
        assert result.exit_code == 0
        assert "MARCEL" in result.output
