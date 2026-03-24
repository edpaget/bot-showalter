from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.batting_stats import BattingStats
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.marcel import MarcelModel
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from tests.helpers import seed_player

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

runner = CliRunner()

pytestmark = pytest.mark.usefixtures("isolated_model_registry")


def _ensure_marcel_registered() -> None:
    """Re-register marcel so each test starts with a known state."""
    register("marcel")(MarcelModel)


class TestListCommand:
    def test_list_shows_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "marcel" in result.output

    def test_list_empty_registry(self) -> None:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No models registered" in result.output


class TestInfoCommand:
    def test_info_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["info", "marcel"])
        assert result.exit_code == 0
        assert "marcel" in result.output
        assert "prepare" in result.output
        assert "predict" in result.output
        assert "evaluate" in result.output

    def test_info_unknown_model(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["info", "nonexistent"])
        assert result.exit_code != 0


class TestFeaturesCommand:
    def test_features_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["features", "marcel"])
        assert result.exit_code == 0
        assert "Features for model 'marcel'" in result.output
        assert "80 features" in result.output
        assert "hr_1" in result.output
        assert "age" in result.output

    def test_features_unknown_model(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["features", "nonexistent"])
        assert result.exit_code != 0

    def test_features_shows_lag(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["features", "marcel"])
        assert "lag=" in result.output

    def test_features_shows_computed(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["features", "marcel"])
        assert "computed=age" in result.output


class TestImportCommand:
    def test_import_command_exists(self) -> None:
        result = runner.invoke(app, ["import", "--help"])
        assert result.exit_code == 0
        assert "third-party" in result.output.lower() or "csv" in result.output.lower()

    def test_import_batting_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        seed_conn = create_connection(db_path)
        seed_player(seed_conn, mlbam_id=545361, fangraphs_id=10155)
        seed_conn.commit()
        seed_conn.close()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        csv_file = tmp_path / "steamer_batting.csv"
        csv_file.write_text("PlayerId,MLBAMID,PA,HR,AVG,WAR\n10155,545361,600,35,0.302,8.5\n")

        result = runner.invoke(
            app,
            [
                "import",
                "steamer",
                str(csv_file),
                "--version",
                "2025.1",
                "--player-type",
                "batter",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(verify_conn))
        projections = proj_repo.get_by_season(2025, system="steamer")
        assert len(projections) == 1
        assert projections[0].source_type == "third_party"
        assert projections[0].stat_json["hr"] == 35
        verify_conn.close()

    def test_import_pitching_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        seed_conn = create_connection(db_path)
        seed_player(seed_conn, mlbam_id=545361, fangraphs_id=10155)
        seed_conn.commit()
        seed_conn.close()

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.factory.create_connection",
            lambda path: create_connection(db_path),
        )

        csv_file = tmp_path / "steamer_pitching.csv"
        csv_file.write_text("PlayerId,MLBAMID,W,L,ERA,SO,IP,WAR\n10155,545361,12,6,3.00,200,185.0,5.5\n")

        result = runner.invoke(
            app,
            [
                "import",
                "steamer",
                str(csv_file),
                "--version",
                "2025.1",
                "--player-type",
                "pitcher",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(verify_conn))
        projections = proj_repo.get_by_season(2025, system="steamer")
        assert len(projections) == 1
        assert projections[0].source_type == "third_party"
        assert projections[0].stat_json["era"] == 3.00
        verify_conn.close()

    def test_import_missing_file_exits_with_error(self) -> None:
        result = runner.invoke(
            app,
            [
                "import",
                "steamer",
                "/nonexistent/path.csv",
                "--version",
                "2025.1",
                "--player-type",
                "batter",
                "--season",
                "2025",
            ],
        )
        assert result.exit_code != 0


def _seed_eval_data(conn: sqlite3.Connection, system: str = "steamer", version: str = "2025.1") -> None:
    """Seed projections and actuals for eval/compare tests."""
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (2, 'Aaron', 'Judge', '1992-04-26', 'R')"
    )
    proj_repo = SqliteProjectionRepo(SingleConnectionProvider(conn))
    batting_repo = SqliteBattingStatsRepo(SingleConnectionProvider(conn))
    proj_repo.upsert(
        Projection(
            player_id=1,
            season=2025,
            system=system,
            version=version,
            player_type=PlayerType.BATTER,
            stat_json={"hr": 30, "avg": 0.280},
            source_type="third_party",
        )
    )
    proj_repo.upsert(
        Projection(
            player_id=2,
            season=2025,
            system=system,
            version=version,
            player_type=PlayerType.BATTER,
            stat_json={"hr": 45, "avg": 0.310},
            source_type="third_party",
        )
    )
    batting_repo.upsert(BattingStats(player_id=1, season=2025, source="fangraphs", hr=28, avg=0.265))
    batting_repo.upsert(BattingStats(player_id=2, season=2025, source="fangraphs", hr=40, avg=0.300))
    conn.commit()


def _seed_better_system(conn: sqlite3.Connection) -> None:
    """Seed a 'better' system that predicts closer to actuals than steamer."""
    proj_repo = SqliteProjectionRepo(SingleConnectionProvider(conn))
    # Closer to actuals: Trout actual=28/0.265, Judge actual=40/0.300
    proj_repo.upsert(
        Projection(
            player_id=1,
            season=2025,
            system="better",
            version="v2",
            player_type=PlayerType.BATTER,
            stat_json={"hr": 28, "avg": 0.265},
            source_type="third_party",
        )
    )
    proj_repo.upsert(
        Projection(
            player_id=2,
            season=2025,
            system="better",
            version="v2",
            player_type=PlayerType.BATTER,
            stat_json={"hr": 40, "avg": 0.300},
            source_type="third_party",
        )
    )
    conn.commit()


def _seed_worse_system(conn: sqlite3.Connection) -> None:
    """Seed a 'worse' system that predicts farther from actuals than steamer."""
    proj_repo = SqliteProjectionRepo(SingleConnectionProvider(conn))
    # Farther from actuals: Trout actual=28/0.265, Judge actual=40/0.300
    proj_repo.upsert(
        Projection(
            player_id=1,
            season=2025,
            system="worse",
            version="v2",
            player_type=PlayerType.BATTER,
            stat_json={"hr": 10, "avg": 0.180},
            source_type="third_party",
        )
    )
    proj_repo.upsert(
        Projection(
            player_id=2,
            season=2025,
            system="worse",
            version="v2",
            player_type=PlayerType.BATTER,
            stat_json={"hr": 15, "avg": 0.200},
            source_type="third_party",
        )
    )
    conn.commit()


class TestCompareCommand:
    def test_compare_command_exists(self) -> None:
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0

    def test_compare_with_tail_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_eval_data(db_conn, system="steamer", version="2025.1")
        _seed_eval_data(db_conn, system="zips", version="2025.1")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["compare", "steamer/2025.1", "zips/2025.1", "--season", "2025", "--tail"],
        )
        assert result.exit_code == 0, result.output

    def test_compare_with_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_eval_data(db_conn, system="steamer", version="2025.1")
        _seed_eval_data(db_conn, system="zips", version="2025.1")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["compare", "steamer/2025.1", "zips/2025.1", "--season", "2025"],
        )
        assert result.exit_code == 0, result.output
        assert "steamer" in result.output
        assert "zips" in result.output

    def test_check_passes_on_improvement(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_eval_data(db_conn, system="steamer", version="2025.1")
        _seed_better_system(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["compare", "steamer/2025.1", "better/v2", "--season", "2025", "--check"],
        )
        assert result.exit_code == 0, result.output
        assert "PASS" in result.output

    def test_check_fails_on_regression(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_eval_data(db_conn, system="steamer", version="2025.1")
        _seed_worse_system(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["compare", "steamer/2025.1", "worse/v2", "--season", "2025", "--check"],
        )
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_check_requires_two_systems(self) -> None:
        result = runner.invoke(
            app,
            ["compare", "steamer/2025.1", "--season", "2025", "--check"],
        )
        assert result.exit_code == 1
        assert "requires exactly 2 systems" in result.output
