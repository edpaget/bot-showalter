from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo

if TYPE_CHECKING:
    import sqlite3

    import pytest

runner = CliRunner()


def _seed_report_data(
    conn: sqlite3.Connection,
    system: str = "statcast-gbm",
    version: str = "latest",
) -> None:
    """Seed players, projections, and batting actuals for report commands."""
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (2, 'Aaron', 'Judge', '1992-04-26', 'R')"
    )
    proj_repo = SqliteProjectionRepo(conn)
    batting_repo = SqliteBattingStatsRepo(conn)
    proj_repo.upsert(
        Projection(
            player_id=1,
            season=2025,
            system=system,
            version=version,
            player_type="batter",
            stat_json={"avg": 0.280, "obp": 0.350},
        )
    )
    proj_repo.upsert(
        Projection(
            player_id=2,
            season=2025,
            system=system,
            version=version,
            player_type="batter",
            stat_json={"avg": 0.300, "obp": 0.370},
        )
    )
    # Player 1 outperforms on avg, player 2 underperforms
    batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", avg=0.310, obp=0.380))
    batting_repo.upsert(BattingStats(player_id=2, season=2025, source="fangraphs", avg=0.270, obp=0.340))
    conn.commit()


class TestReportCommands:
    def test_overperformers_command_exists(self) -> None:
        result = runner.invoke(app, ["report", "overperformers", "--help"])
        assert result.exit_code == 0

    def test_underperformers_command_exists(self) -> None:
        result = runner.invoke(app, ["report", "underperformers", "--help"])
        assert result.exit_code == 0

    def test_overperformers_with_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_report_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            [
                "report",
                "overperformers",
                "statcast-gbm/latest",
                "--season",
                "2025",
                "--player-type",
                "batter",
                "--stat",
                "avg",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Overperformers" in result.output
        assert "Mike Trout" in result.output or "Trout" in result.output

    def test_underperformers_with_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_report_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            [
                "report",
                "underperformers",
                "statcast-gbm/latest",
                "--season",
                "2025",
                "--player-type",
                "batter",
                "--stat",
                "avg",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Underperformers" in result.output
        assert "Aaron Judge" in result.output or "Judge" in result.output
