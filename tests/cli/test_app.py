import sqlite3

import pytest
from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.models.marcel import MarcelModel
from fantasy_baseball_manager.models.registry import _clear, register

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
