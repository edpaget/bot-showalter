import sqlite3

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.projection_lookup import ProjectionLookupService


def _seed_player(conn: sqlite3.Connection, name_first: str, name_last: str, mlbam_id: int) -> int:
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first=name_first, name_last=name_last, mlbam_id=mlbam_id))


def _seed_projection(
    conn: sqlite3.Connection,
    player_id: int,
    season: int = 2025,
    system: str = "steamer",
    version: str = "2025.1",
    player_type: str = "batter",
    stat_json: dict | None = None,
    source_type: str = "third_party",
) -> None:
    repo = SqliteProjectionRepo(conn)
    repo.upsert(
        Projection(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            player_type=player_type,
            stat_json=stat_json or {"hr": 30, "avg": 0.280},
            source_type=source_type,
        )
    )


def _make_service(conn: sqlite3.Connection) -> ProjectionLookupService:
    return ProjectionLookupService(SqlitePlayerRepo(conn), SqliteProjectionRepo(conn))


class TestLookup:
    def test_match_by_last_name(self, conn: sqlite3.Connection) -> None:
        pid = _seed_player(conn, "Mike", "Trout", 545361)
        _seed_projection(conn, pid)
        svc = _make_service(conn)

        results = svc.lookup("Trout", 2025)
        assert len(results) == 1
        assert results[0].player_name == "Mike Trout"
        assert results[0].system == "steamer"
        assert results[0].stats["hr"] == 30

    def test_case_insensitive(self, conn: sqlite3.Connection) -> None:
        pid = _seed_player(conn, "Mike", "Trout", 545361)
        _seed_projection(conn, pid)
        svc = _make_service(conn)

        results = svc.lookup("trout", 2025)
        assert len(results) == 1

    def test_no_match_returns_empty(self, conn: sqlite3.Connection) -> None:
        svc = _make_service(conn)
        results = svc.lookup("Nobody", 2025)
        assert results == []

    def test_system_filter(self, conn: sqlite3.Connection) -> None:
        pid = _seed_player(conn, "Mike", "Trout", 545361)
        _seed_projection(conn, pid, system="steamer", version="2025.1")
        _seed_projection(conn, pid, system="zips", version="2025.1")
        svc = _make_service(conn)

        results = svc.lookup("Trout", 2025, system="steamer")
        assert len(results) == 1
        assert results[0].system == "steamer"

    def test_last_first_format_disambiguates(self, conn: sqlite3.Connection) -> None:
        pid_joe = _seed_player(conn, "Joe", "Smith", 100001)
        pid_john = _seed_player(conn, "John", "Smith", 100002)
        _seed_projection(conn, pid_joe)
        _seed_projection(conn, pid_john)
        svc = _make_service(conn)

        results = svc.lookup("Smith, Joe", 2025)
        assert len(results) == 1
        assert results[0].player_name == "Joe Smith"

    def test_player_with_no_projections_excluded(self, conn: sqlite3.Connection) -> None:
        _seed_player(conn, "Mike", "Trout", 545361)
        svc = _make_service(conn)

        results = svc.lookup("Trout", 2025)
        assert results == []


class TestListSystems:
    def test_returns_systems_for_season(self, conn: sqlite3.Connection) -> None:
        pid = _seed_player(conn, "Mike", "Trout", 545361)
        _seed_projection(conn, pid, system="steamer", version="2025.1")
        _seed_projection(conn, pid, system="zips", version="2025.1")
        svc = _make_service(conn)

        summaries = svc.list_systems(2025)
        assert len(summaries) == 2
        systems = {s.system for s in summaries}
        assert systems == {"steamer", "zips"}

    def test_counts_batters_and_pitchers(self, conn: sqlite3.Connection) -> None:
        pid1 = _seed_player(conn, "Mike", "Trout", 545361)
        pid2 = _seed_player(conn, "Shohei", "Ohtani", 660271)
        _seed_projection(conn, pid1, system="steamer", player_type="batter")
        _seed_projection(conn, pid2, system="steamer", player_type="pitcher")
        svc = _make_service(conn)

        summaries = svc.list_systems(2025)
        assert len(summaries) == 1
        assert summaries[0].batter_count == 1
        assert summaries[0].pitcher_count == 1

    def test_empty_season_returns_empty(self, conn: sqlite3.Connection) -> None:
        svc = _make_service(conn)
        summaries = svc.list_systems(2025)
        assert summaries == []

    def test_different_versions_listed_separately(self, conn: sqlite3.Connection) -> None:
        pid = _seed_player(conn, "Mike", "Trout", 545361)
        _seed_projection(conn, pid, system="steamer", version="2025.1")
        _seed_projection(conn, pid, system="steamer", version="2025.2")
        svc = _make_service(conn)

        summaries = svc.list_systems(2025)
        assert len(summaries) == 2
        versions = {s.version for s in summaries}
        assert versions == {"2025.1", "2025.2"}
