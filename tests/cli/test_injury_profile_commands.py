from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import ILStint
from fantasy_baseball_manager.repos.il_stint_repo import SqliteILStintRepo
from tests.helpers import seed_player

if TYPE_CHECKING:
    import sqlite3

    import pytest

runner = CliRunner()


def _seed_il_data(conn: sqlite3.Connection) -> None:
    """Seed players and IL stints for testing injury commands."""
    seed_player(conn, player_id=1, name_first="Mike", name_last="Trout")
    seed_player(conn, player_id=2, name_first="Aaron", name_last="Judge")
    seed_player(conn, player_id=3, name_first="Healthy", name_last="Player")

    il_repo = SqliteILStintRepo(SingleConnectionProvider(conn))
    # Mike Trout: chronically injured
    il_repo.upsert(
        ILStint(player_id=1, season=2023, start_date="2023-04-15", il_type="10-day", days=15, injury_location="calf")
    )
    il_repo.upsert(
        ILStint(player_id=1, season=2023, start_date="2023-07-01", il_type="60-day", days=90, injury_location="knee")
    )
    il_repo.upsert(
        ILStint(player_id=1, season=2024, start_date="2024-05-10", il_type="10-day", days=20, injury_location="calf")
    )

    # Aaron Judge: one incident
    il_repo.upsert(
        ILStint(player_id=2, season=2024, start_date="2024-06-01", il_type="10-day", days=12, injury_location="hip")
    )
    conn.commit()


class TestInjuryProfileCommand:
    def test_help(self) -> None:
        result = runner.invoke(app, ["report", "injury-profile", "--help"])
        assert result.exit_code == 0

    def test_player_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "injury-profile", "Trout", "--seasons", "3", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Injury Profile" in result.output
        assert "Mike Trout" in result.output
        assert "calf" in result.output or "knee" in result.output

    def test_player_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "injury-profile", "Nobody", "--data-dir", "./data"],
        )
        assert result.exit_code == 1
        assert "no player found" in result.output

    def test_healthy_player_clean_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "injury-profile", "Healthy Player", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Total IL stints:      0" in result.output
        assert "Total days lost:      0" in result.output


class TestInjuryRisksCommand:
    def test_help(self) -> None:
        result = runner.invoke(app, ["report", "injury-risks", "--help"])
        assert result.exit_code == 0

    def test_leaderboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "injury-risks", "--season", "2024", "--seasons-back", "3", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Injury Risk Leaderboard" in result.output
        assert "Mike Trout" in result.output or "Trout" in result.output

    def test_leaderboard_min_stints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            [
                "report",
                "injury-risks",
                "--season",
                "2024",
                "--min-stints",
                "3",
                "--seasons-back",
                "3",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        # Only Trout has 3+ stints
        assert "Trout" in result.output
        assert "Judge" not in result.output

    def test_leaderboard_top_n(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            [
                "report",
                "injury-risks",
                "--season",
                "2024",
                "--top",
                "1",
                "--seasons-back",
                "3",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        # Only top 1 should show (Trout, most days lost)
        assert "Trout" in result.output

    def test_empty_leaderboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        # Seed players but no IL stints
        seed_player(db_conn, player_id=1, name_first="Test", name_last="Player")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "injury-risks", "--season", "2024", "--data-dir", "./data"],
        )
        assert result.exit_code == 0
        assert "No injury-prone players found" in result.output


class TestInjuryEstimateCommand:
    def test_help(self) -> None:
        result = runner.invoke(app, ["report", "injury-estimate", "--help"])
        assert result.exit_code == 0

    def test_player_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "injury-estimate", "Trout", "--season", "2026", "--seasons-back", "5", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Injury Estimate" in result.output
        assert "Mike Trout" in result.output
        assert "Expected days lost" in result.output
        assert "P(full season)" in result.output

    def test_player_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "injury-estimate", "Nobody", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 1
        assert "no player found" in result.output

    def test_healthy_player_nonzero_baseline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "injury-estimate", "Healthy Player", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        # Even healthy players should have nonzero baseline
        assert "Expected days lost" in result.output


class TestGamesLostCommand:
    def test_help(self) -> None:
        result = runner.invoke(app, ["report", "games-lost", "--help"])
        assert result.exit_code == 0

    def test_leaderboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "games-lost", "--season", "2026", "--seasons-back", "5", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "Expected Games Lost" in result.output
        assert "Trout" in result.output

    def test_leaderboard_top_n(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_il_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "games-lost", "--season", "2026", "--top", "1", "--seasons-back", "5", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        # Top 1 should show the most injury-prone player
        assert "Trout" in result.output

    def test_empty_leaderboard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        seed_player(db_conn, player_id=1, name_first="Test", name_last="Player")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["report", "games-lost", "--season", "2026", "--data-dir", "./data"],
        )
        assert result.exit_code == 0
        assert "No players found" in result.output


class TestInjuryAdjustedValuesCommand:
    def test_help(self) -> None:
        result = runner.invoke(app, ["report", "injury-adjusted-values", "--help"])
        assert result.exit_code == 0
        assert "injury" in result.output.lower()
