import textwrap
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.cli.factory import KeeperContext
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.repos.keeper_repo import SqliteKeeperCostRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from tests.helpers import seed_player

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator

    import pytest

runner = CliRunner()


def _build_test_keeper_context(conn: sqlite3.Connection) -> Any:
    @contextmanager
    def _ctx(data_dir: str) -> Iterator[KeeperContext]:
        yield KeeperContext(
            conn=conn,
            keeper_repo=SqliteKeeperCostRepo(conn),
            player_repo=SqlitePlayerRepo(conn),
        )

    return _ctx


class TestKeeperImport:
    def test_import_success(self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Trout")
        seed_player(conn, name_first="Shohei", name_last="Ohtani")

        csv_file = tmp_path / "keepers.csv"  # type: ignore[operator]
        csv_file.write_text(
            textwrap.dedent("""\
            Player,Cost,Years
            Mike Trout,25,2
            Shohei Ohtani,15,1
        """)
        )

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(app, ["keeper", "import", str(csv_file), "--season", "2026", "--league", "dynasty"])
        assert result.exit_code == 0, result.output
        assert "Loaded 2" in result.output

        repo = SqliteKeeperCostRepo(conn)
        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 2
        conn.close()

    def test_import_file_not_found(self) -> None:
        result = runner.invoke(
            app, ["keeper", "import", "/nonexistent/file.csv", "--season", "2026", "--league", "dynasty"]
        )
        assert result.exit_code == 1
        assert "file not found" in result.output

    def test_import_idempotent(self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Trout")

        csv_file = tmp_path / "keepers.csv"  # type: ignore[operator]
        csv_file.write_text(
            textwrap.dedent("""\
            Player,Cost
            Mike Trout,25
        """)
        )

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        # Import twice
        runner.invoke(app, ["keeper", "import", str(csv_file), "--season", "2026", "--league", "dynasty"])
        result = runner.invoke(app, ["keeper", "import", str(csv_file), "--season", "2026", "--league", "dynasty"])
        assert result.exit_code == 0, result.output

        repo = SqliteKeeperCostRepo(conn)
        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 1
        conn.close()


class TestKeeperSet:
    def test_set_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Trout")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app, ["keeper", "set", "Trout", "--cost", "25", "--season", "2026", "--league", "dynasty"]
        )
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "$25" in result.output

        repo = SqliteKeeperCostRepo(conn)
        stored = repo.find_by_season_league(2026, "dynasty")
        assert len(stored) == 1
        assert stored[0].cost == 25.0
        conn.close()

    def test_set_player_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app, ["keeper", "set", "Nobody", "--cost", "10", "--season", "2026", "--league", "dynasty"]
        )
        assert result.exit_code == 1
        assert "no player found" in result.output

    def test_set_ambiguous_player(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = create_connection(":memory:")
        seed_player(conn, name_first="Mike", name_last="Smith")
        seed_player(conn, name_first="John", name_last="Smith")

        monkeypatch.setattr(
            "fantasy_baseball_manager.cli.commands.keeper.build_keeper_context",
            _build_test_keeper_context(conn),
        )

        result = runner.invoke(
            app, ["keeper", "set", "Smith", "--cost", "10", "--season", "2026", "--league", "dynasty"]
        )
        assert result.exit_code == 1
        assert "ambiguous" in result.output
        conn.close()
