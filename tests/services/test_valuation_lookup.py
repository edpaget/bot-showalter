import sqlite3

from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.valuation_repo import SqliteValuationRepo
from fantasy_baseball_manager.services.valuation_lookup import ValuationLookupService
from tests.helpers import seed_player


def _seed_valuation(
    conn: sqlite3.Connection,
    player_id: int,
    season: int = 2025,
    system: str = "zar",
    version: str = "1.0",
    projection_system: str = "steamer",
    projection_version: str = "2025.1",
    player_type: str = "batter",
    position: str = "OF",
    value: float = 25.0,
    rank: int = 1,
    category_scores: dict[str, float] | None = None,
) -> None:
    repo = SqliteValuationRepo(conn)
    repo.upsert(
        Valuation(
            player_id=player_id,
            season=season,
            system=system,
            version=version,
            projection_system=projection_system,
            projection_version=projection_version,
            player_type=player_type,
            position=position,
            value=value,
            rank=rank,
            category_scores=category_scores or {"hr": 2.1, "sb": 0.5},
        )
    )


def _make_service(conn: sqlite3.Connection) -> ValuationLookupService:
    return ValuationLookupService(SqlitePlayerRepo(conn), SqliteValuationRepo(conn))


class TestLookup:
    def test_match_by_last_name(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid, value=42.5, rank=1)
        svc = _make_service(conn)

        results = svc.lookup("Soto", 2025)
        assert len(results) == 1
        assert results[0].player_name == "Juan Soto"
        assert results[0].system == "zar"
        assert results[0].version == "1.0"
        assert results[0].projection_system == "steamer"
        assert results[0].projection_version == "2025.1"
        assert results[0].player_type == "batter"
        assert results[0].position == "OF"
        assert results[0].value == 42.5
        assert results[0].rank == 1
        assert results[0].category_scores == {"hr": 2.1, "sb": 0.5}

    def test_case_insensitive(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid)
        svc = _make_service(conn)

        results = svc.lookup("soto", 2025)
        assert len(results) == 1

    def test_no_match_returns_empty(self, conn: sqlite3.Connection) -> None:
        svc = _make_service(conn)
        results = svc.lookup("Nobody", 2025)
        assert results == []

    def test_system_filter(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid, system="zar", version="1.0")
        _seed_valuation(conn, pid, system="auction", version="1.0")
        svc = _make_service(conn)

        results = svc.lookup("Soto", 2025, system="zar")
        assert len(results) == 1
        assert results[0].system == "zar"

    def test_last_first_format_disambiguates(self, conn: sqlite3.Connection) -> None:
        pid_joe = seed_player(conn, name_first="Joe", name_last="Smith", mlbam_id=100001)
        pid_john = seed_player(conn, name_first="John", name_last="Smith", mlbam_id=100002)
        _seed_valuation(conn, pid_joe)
        _seed_valuation(conn, pid_john)
        svc = _make_service(conn)

        results = svc.lookup("Smith, Joe", 2025)
        assert len(results) == 1
        assert results[0].player_name == "Joe Smith"

    def test_player_with_no_valuations_excluded(self, conn: sqlite3.Connection) -> None:
        seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        svc = _make_service(conn)

        results = svc.lookup("Soto", 2025)
        assert results == []

    def test_multiple_valuations_for_player(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid, system="zar", version="1.0", value=42.5)
        _seed_valuation(conn, pid, system="auction", version="1.0", value=38.0)
        svc = _make_service(conn)

        results = svc.lookup("Soto", 2025)
        assert len(results) == 2
        systems = {r.system for r in results}
        assert systems == {"zar", "auction"}


class TestRankings:
    def test_returns_ranked_list(self, conn: sqlite3.Connection) -> None:
        pid1 = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid2 = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        _seed_valuation(conn, pid1, rank=1, value=42.5)
        _seed_valuation(conn, pid2, rank=2, value=38.0)
        svc = _make_service(conn)

        results = svc.rankings(2025)
        assert len(results) == 2
        assert results[0].rank == 1
        assert results[0].player_name == "Juan Soto"
        assert results[1].rank == 2
        assert results[1].player_name == "Aaron Judge"

    def test_filter_by_system(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid, system="zar", rank=1)
        _seed_valuation(conn, pid, system="auction", rank=1)
        svc = _make_service(conn)

        results = svc.rankings(2025, system="zar")
        assert len(results) == 1
        assert results[0].system == "zar"

    def test_filter_by_player_type(self, conn: sqlite3.Connection) -> None:
        pid1 = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid2 = seed_player(conn, name_first="Gerrit", name_last="Cole", mlbam_id=543037)
        _seed_valuation(conn, pid1, player_type="batter", rank=1)
        _seed_valuation(conn, pid2, player_type="pitcher", rank=2)
        svc = _make_service(conn)

        results = svc.rankings(2025, player_type="batter")
        assert len(results) == 1
        assert results[0].player_name == "Juan Soto"

    def test_filter_by_position(self, conn: sqlite3.Connection) -> None:
        pid1 = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid2 = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        _seed_valuation(conn, pid1, position="OF", rank=1)
        _seed_valuation(conn, pid2, position="DH", rank=2)
        svc = _make_service(conn)

        results = svc.rankings(2025, position="OF")
        assert len(results) == 1
        assert results[0].player_name == "Juan Soto"

    def test_top_limits_results(self, conn: sqlite3.Connection) -> None:
        pid1 = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        pid2 = seed_player(conn, name_first="Aaron", name_last="Judge", mlbam_id=592450)
        pid3 = seed_player(conn, name_first="Shohei", name_last="Ohtani", mlbam_id=660271)
        _seed_valuation(conn, pid1, rank=1, value=42.5)
        _seed_valuation(conn, pid2, rank=2, value=38.0)
        _seed_valuation(conn, pid3, rank=3, value=35.0)
        svc = _make_service(conn)

        results = svc.rankings(2025, top=2)
        assert len(results) == 2
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_empty_season_returns_empty(self, conn: sqlite3.Connection) -> None:
        svc = _make_service(conn)
        results = svc.rankings(2025)
        assert results == []

    def test_player_name_resolved(self, conn: sqlite3.Connection) -> None:
        pid = seed_player(conn, name_first="Juan", name_last="Soto", mlbam_id=665742)
        _seed_valuation(conn, pid, rank=1)
        svc = _make_service(conn)

        results = svc.rankings(2025)
        assert len(results) == 1
        assert results[0].player_name == "Juan Soto"
