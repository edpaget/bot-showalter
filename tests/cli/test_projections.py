from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo

if TYPE_CHECKING:
    import sqlite3

    import pytest

runner = CliRunner()


def _seed_projections_data(conn: sqlite3.Connection, system: str = "steamer", version: str = "2025.1") -> None:
    """Seed player and projection data for projections commands."""
    player_repo = SqlitePlayerRepo(conn)
    proj_repo = SqliteProjectionRepo(conn)
    pid = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
    proj_repo.upsert(
        Projection(
            player_id=pid,
            season=2025,
            system=system,
            version=version,
            player_type="batter",
            stat_json={"hr": 30, "avg": 0.280},
            source_type="third_party",
        )
    )
    conn.commit()


class TestProjectionsLookupCommand:
    def test_lookup_returns_stats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_projections_data(db_conn)
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["projections", "lookup", "Trout", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "Mike Trout" in result.output
        assert "steamer" in result.output
        assert "hr" in result.output

    def test_lookup_system_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_projections_data(db_conn, system="steamer")
        _seed_projections_data(db_conn, system="zips")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(
            app,
            ["projections", "lookup", "Trout", "--season", "2025", "--system", "steamer", "--data-dir", "./data"],
        )
        assert result.exit_code == 0, result.output
        assert "steamer" in result.output
        assert "zips" not in result.output

    def test_lookup_no_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["projections", "lookup", "Nobody", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No projections found" in result.output


class TestProjectionsSystemsCommand:
    def test_systems_lists_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        _seed_projections_data(db_conn, system="steamer")
        _seed_projections_data(db_conn, system="zips")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["projections", "systems", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "steamer" in result.output
        assert "zips" in result.output

    def test_systems_empty_season(self, monkeypatch: pytest.MonkeyPatch) -> None:
        db_conn = create_connection(":memory:")
        monkeypatch.setattr("fantasy_baseball_manager.cli.factory.create_connection", lambda path: db_conn)

        result = runner.invoke(app, ["projections", "systems", "--season", "2025", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output
        assert "No projection systems found" in result.output
