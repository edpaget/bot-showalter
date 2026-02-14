import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.models.marcel import MarcelModel
from fantasy_baseball_manager.models.registry import _clear, register
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo

runner = CliRunner()


def _ensure_marcel_registered() -> None:
    """Clear and re-register marcel so each test starts with a known state."""
    _clear()
    register("marcel")(MarcelModel)


def _seed_batting_data(conn: sqlite3.Connection) -> None:
    """Insert minimal data for prepare integration test."""
    conn.execute(
        "INSERT INTO player (id, name_first, name_last, birth_date, bats) "
        "VALUES (1, 'Mike', 'Trout', '1991-08-07', 'R')"
    )
    batting_rows = [
        (1, 2020, "fangraphs", 250, 17, 35, 200, 56),
        (1, 2021, "fangraphs", 500, 30, 60, 420, 120),
        (1, 2022, "fangraphs", 550, 40, 70, 460, 140),
        (1, 2023, "fangraphs", 600, 35, 65, 500, 150),
    ]
    conn.executemany(
        "INSERT INTO batting_stats (player_id, season, source, pa, hr, bb, ab, h) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        batting_rows,
    )
    conn.commit()


class TestListCommand:
    def test_list_shows_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "marcel" in result.output

    def test_list_empty_registry(self) -> None:
        _clear()
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
        assert "train" in result.output
        assert "evaluate" in result.output

    def test_info_unknown_model(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["info", "nonexistent"])
        assert result.exit_code != 0


class TestActionCommands:
    def test_train_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["train", "marcel"])
        assert result.exit_code == 0
        assert "marcel" in result.output.lower()

    def test_prepare_marcel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ensure_marcel_registered()
        seeded_conn = create_connection(":memory:")
        _seed_batting_data(seeded_conn)
        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.app.create_connection",
            lambda path: seeded_conn,
        )
        result = runner.invoke(app, ["prepare", "marcel", "--season", "2023"])
        assert result.exit_code == 0

    def test_evaluate_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["evaluate", "marcel"])
        assert result.exit_code == 0

    def test_finetune_marcel_fails(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["finetune", "marcel"])
        assert result.exit_code != 0
        assert "does not support" in result.output.lower()

    def test_predict_marcel_fails(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["predict", "marcel"])
        assert result.exit_code != 0

    def test_ablate_marcel_fails(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["ablate", "marcel"])
        assert result.exit_code != 0

    def test_train_with_output_dir(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["train", "marcel", "--output-dir", "/tmp/out"])
        assert result.exit_code == 0

    def test_train_with_seasons(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["train", "marcel", "--season", "2023", "--season", "2024"])
        assert result.exit_code == 0


class TestFeaturesCommand:
    def test_features_marcel(self) -> None:
        _ensure_marcel_registered()
        result = runner.invoke(app, ["features", "marcel"])
        assert result.exit_code == 0
        assert "Features for model 'marcel'" in result.output
        assert "16 features" in result.output
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


def _seed_player_for_import(conn: sqlite3.Connection) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361, fangraphs_id=10155))


class TestImportCommand:
    def test_import_command_exists(self) -> None:
        result = runner.invoke(app, ["import", "--help"])
        assert result.exit_code == 0
        assert "third-party" in result.output.lower() or "csv" in result.output.lower()

    def test_import_batting_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_player_for_import(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.app.create_connection", lambda path: db_conn)

        csv_file = tmp_path / "steamer_batting.csv"
        csv_file.write_text("playerid,PA,HR,AVG,WAR\n10155,600,35,0.302,8.5\n")

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

        proj_repo = SqliteProjectionRepo(db_conn)
        projections = proj_repo.get_by_season(2025, system="steamer")
        assert len(projections) == 1
        assert projections[0].source_type == "third_party"
        assert projections[0].stat_json["hr"] == 35

    def test_import_pitching_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_player_for_import(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.app.create_connection", lambda path: db_conn)

        csv_file = tmp_path / "steamer_pitching.csv"
        csv_file.write_text("playerid,W,L,ERA,SO,IP,WAR\n10155,12,6,3.00,200,185.0,5.5\n")

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

        proj_repo = SqliteProjectionRepo(db_conn)
        projections = proj_repo.get_by_season(2025, system="steamer")
        assert len(projections) == 1
        assert projections[0].source_type == "third_party"
        assert projections[0].stat_json["era"] == 3.00

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
