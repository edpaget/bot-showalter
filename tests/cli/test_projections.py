from typing import TYPE_CHECKING, Any

import httpx
from typer.testing import CliRunner

from fantasy_baseball_manager.cli.app import app
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.ingest.fangraphs_projection_source import FgProjectionSource
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from tests.helpers import seed_player

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

    import pytest

runner = CliRunner()


def _seed_projections_data(conn: sqlite3.Connection, system: str = "steamer", version: str = "2025.1") -> None:
    """Seed player and projection data for projections commands."""
    player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
    proj_repo = SqliteProjectionRepo(SingleConnectionProvider(conn))
    pid = player_repo.upsert(Player(name_first="Mike", name_last="Trout", mlbam_id=545361))
    proj_repo.upsert(
        Projection(
            player_id=pid,
            season=2025,
            system=system,
            version=version,
            player_type=PlayerType.BATTER,
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


# --- Canned API data for sync tests ---

_BATTING_ROW: dict[str, Any] = {
    "playerid": "10155",
    "xMLBAMID": 545361,
    "PlayerName": "Mike Trout",
    "PA": 600,
    "AB": 530,
    "H": 160,
    "2B": 30,
    "3B": 5,
    "HR": 35,
    "RBI": 90,
    "R": 100,
    "SB": 15,
    "CS": 3,
    "BB": 60,
    "SO": 120,
    "AVG": 0.302,
    "OBP": 0.395,
    "SLG": 0.575,
    "OPS": 0.970,
    "wOBA": 0.410,
    "wRC+": 170.0,
    "WAR": 8.5,
}

_PITCHING_ROW: dict[str, Any] = {
    "playerid": "19755",
    "xMLBAMID": 669373,
    "PlayerName": "Corbin Burnes",
    "W": 15,
    "L": 5,
    "G": 30,
    "GS": 30,
    "SV": 0,
    "IP": 180.0,
    "H": 120,
    "ER": 50,
    "HR": 15,
    "BB": 40,
    "SO": 200,
    "ERA": 2.80,
    "WHIP": 0.95,
    "K/9": 10.0,
    "BB/9": 2.0,
    "FIP": 2.90,
    "WAR": 6.0,
}


class _FakeTransport(httpx.BaseTransport):
    """Returns canned JSON for any request URL and records requests."""

    def __init__(self, batting_rows: list[dict[str, Any]], pitching_rows: list[dict[str, Any]]) -> None:
        self._batting = batting_rows
        self._pitching = pitching_rows
        self.requests: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        stats = dict(request.url.params).get("stats", "bat")
        data = self._pitching if stats == "pit" else self._batting
        return httpx.Response(200, json=data)


def _seed_sync_db(db_path: Path) -> None:
    """Create and seed a file-based DB with test players."""
    conn = create_connection(db_path)
    seed_player(conn, fangraphs_id=10155, mlbam_id=545361, name_first="Mike", name_last="Trout")
    seed_player(conn, fangraphs_id=19755, mlbam_id=669373, name_first="Corbin", name_last="Burnes")
    conn.commit()
    conn.close()


def _patch_sync_transport(
    monkeypatch: pytest.MonkeyPatch,
    db_path: Path,
    batting_rows: list[dict[str, Any]] | None = None,
    pitching_rows: list[dict[str, Any]] | None = None,
) -> _FakeTransport:
    """Monkeypatch create_connection to use file DB and inject fake HTTP transport."""
    monkeypatch.setattr(
        "fantasy_baseball_manager.cli.factory.create_connection",
        lambda path: create_connection(db_path),
    )
    transport = _FakeTransport(batting_rows or [_BATTING_ROW], pitching_rows or [_PITCHING_ROW])
    real_init = FgProjectionSource.__init__

    def _patched_init(self: FgProjectionSource, *args: Any, **kwargs: Any) -> None:
        kwargs["client"] = httpx.Client(transport=transport)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(
        "fantasy_baseball_manager.cli.commands.projections.FgProjectionSource.__init__",
        _patched_init,
    )
    return transport


class TestProjectionsSyncCommand:
    def test_sync_single_system(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        _seed_sync_db(db_path)
        _patch_sync_transport(monkeypatch, db_path)

        result = runner.invoke(app, ["projections", "sync", "fangraphs-dc", "--season", "2026", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(verify_conn))
        projs = proj_repo.get_by_season(2026, system="fangraphs-dc")
        assert len(projs) == 2  # 1 batter + 1 pitcher
        types = {p.player_type for p in projs}
        assert types == {"batter", "pitcher"}
        for p in projs:
            assert p.source_type == "third_party"
        verify_conn.close()

    def test_sync_all_systems(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        _seed_sync_db(db_path)
        _patch_sync_transport(monkeypatch, db_path)

        result = runner.invoke(app, ["projections", "sync", "--all", "--season", "2026", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(verify_conn))
        systems = {p.system for p in proj_repo.get_by_season(2026)}
        assert systems == {"fangraphs-dc", "steamer", "zips"}
        verify_conn.close()

    def test_sync_custom_version(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        _seed_sync_db(db_path)
        _patch_sync_transport(monkeypatch, db_path)

        result = runner.invoke(
            app,
            [
                "projections",
                "sync",
                "steamer",
                "--season",
                "2026",
                "--version",
                "2026-preseason",
                "--data-dir",
                "./data",
            ],
        )
        assert result.exit_code == 0, result.output

        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(verify_conn))
        projs = proj_repo.get_by_season(2026, system="steamer")
        assert all(p.version == "2026-preseason" for p in projs)
        verify_conn.close()

    def test_sync_invalid_system(self) -> None:
        result = runner.invoke(app, ["projections", "sync", "bogus", "--season", "2026", "--data-dir", "./data"])
        assert result.exit_code != 0

    def test_sync_requires_system_or_all(self) -> None:
        result = runner.invoke(app, ["projections", "sync", "--season", "2026", "--data-dir", "./data"])
        assert result.exit_code != 0

    def test_sync_help(self) -> None:
        result = runner.invoke(app, ["projections", "sync", "--help"])
        assert result.exit_code == 0
        assert "sync" in result.output.lower() or "season" in result.output.lower()

    def test_sync_ros_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        _seed_sync_db(db_path)
        transport = _patch_sync_transport(monkeypatch, db_path)

        result = runner.invoke(
            app, ["projections", "sync", "steamer", "--season", "2026", "--ros", "--data-dir", "./data"]
        )
        assert result.exit_code == 0, result.output

        # Verify API type used was steamerr (ROS variant)
        api_types = {dict(r.url.params)["type"] for r in transport.requests}
        assert api_types == {"steamerr"}

        # Verify version has -ros suffix
        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(verify_conn))
        projs = proj_repo.get_by_season(2026, system="steamer")
        assert len(projs) == 2
        assert all(p.version.endswith("-ros") for p in projs)
        verify_conn.close()

    def test_sync_ros_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        _seed_sync_db(db_path)
        transport = _patch_sync_transport(monkeypatch, db_path)

        result = runner.invoke(
            app, ["projections", "sync", "--all", "--season", "2026", "--ros", "--data-dir", "./data"]
        )
        assert result.exit_code == 0, result.output

        # Verify ROS API types were used
        api_types = {dict(r.url.params)["type"] for r in transport.requests}
        assert api_types == {"rfangraphsdc", "steamerr", "rzips"}

        # Verify all 3 systems stored with -ros version
        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(verify_conn))
        systems = {p.system for p in proj_repo.get_by_season(2026)}
        assert systems == {"fangraphs-dc", "steamer", "zips"}
        projs = proj_repo.get_by_season(2026)
        assert all(p.version.endswith("-ros") for p in projs)
        verify_conn.close()


class TestProjectionsRefreshCommand:
    def test_refresh_command(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = tmp_path / "fbm.db"
        _seed_sync_db(db_path)
        transport = _patch_sync_transport(monkeypatch, db_path)

        result = runner.invoke(app, ["projections", "refresh", "--season", "2026", "--data-dir", "./data"])
        assert result.exit_code == 0, result.output

        # Verify ROS API types were used
        api_types = {dict(r.url.params)["type"] for r in transport.requests}
        assert api_types == {"rfangraphsdc", "steamerr", "rzips"}

        # Verify all 3 systems stored with -ros version
        verify_conn = create_connection(db_path)
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(verify_conn))
        systems = {p.system for p in proj_repo.get_by_season(2026)}
        assert systems == {"fangraphs-dc", "steamer", "zips"}
        projs = proj_repo.get_by_season(2026)
        assert all(p.version.endswith("-ros") for p in projs)
        verify_conn.close()

    def test_refresh_help(self) -> None:
        result = runner.invoke(app, ["projections", "refresh", "--help"])
        assert result.exit_code == 0
        assert "refresh" in result.output.lower() or "rest-of-season" in result.output.lower()
