import sqlite3

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.repos.adp_repo import SqliteADPRepo
from tests.helpers import seed_player


def _make_adp(player_id: int, **overrides: object) -> ADP:
    defaults: dict[str, object] = {
        "player_id": player_id,
        "season": 2026,
        "provider": "fantasypros",
        "overall_pick": 1.5,
        "rank": 1,
        "positions": "CF,RF,DH",
    }
    defaults.update(overrides)
    return ADP(**defaults)  # type: ignore[arg-type]


class TestSqliteADPRepo:
    def test_upsert_and_get_by_player_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteADPRepo(conn)
        repo.upsert(_make_adp(player_id))
        results = repo.get_by_player_season(player_id, 2026)
        assert len(results) == 1
        assert results[0].player_id == player_id
        assert results[0].provider == "fantasypros"
        assert results[0].overall_pick == 1.5
        assert results[0].rank == 1
        assert results[0].positions == "CF,RF,DH"

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteADPRepo(conn)
        repo.upsert(_make_adp(player_id, overall_pick=2.0, rank=2))
        repo.upsert(_make_adp(player_id, overall_pick=1.0, rank=1))
        results = repo.get_by_player_season(player_id, 2026)
        assert len(results) == 1
        assert results[0].overall_pick == 1.0
        assert results[0].rank == 1

    def test_get_by_season(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteADPRepo(conn)
        repo.upsert(_make_adp(player_id, season=2026))
        repo.upsert(_make_adp(player_id, season=2025))
        results = repo.get_by_season(2026)
        assert len(results) == 1
        assert results[0].season == 2026

    def test_get_by_season_with_provider(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteADPRepo(conn)
        repo.upsert(_make_adp(player_id, provider="espn"))
        repo.upsert(_make_adp(player_id, provider="yahoo"))
        results = repo.get_by_season(2026, provider="espn")
        assert len(results) == 1
        assert results[0].provider == "espn"

    def test_multiple_providers_per_player(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteADPRepo(conn)
        repo.upsert(_make_adp(player_id, provider="espn", overall_pick=3.0, rank=3))
        repo.upsert(_make_adp(player_id, provider="yahoo", overall_pick=5.0, rank=5))
        repo.upsert(_make_adp(player_id, provider="fantasypros", overall_pick=4.0, rank=4))
        results = repo.get_by_player_season(player_id, 2026)
        assert len(results) == 3
        providers = {r.provider for r in results}
        assert providers == {"espn", "yahoo", "fantasypros"}

    def test_as_of_none_round_trips(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteADPRepo(conn)
        repo.upsert(_make_adp(player_id, as_of=None))
        results = repo.get_by_player_season(player_id, 2026)
        assert len(results) == 1
        assert results[0].as_of is None

    def test_as_of_with_date(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteADPRepo(conn)
        repo.upsert(_make_adp(player_id, as_of="2026-03-01"))
        results = repo.get_by_player_season(player_id, 2026)
        assert len(results) == 1
        assert results[0].as_of == "2026-03-01"

    def test_as_of_creates_separate_records(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteADPRepo(conn)
        repo.upsert(_make_adp(player_id, as_of=None, overall_pick=5.0))
        repo.upsert(_make_adp(player_id, as_of="2026-03-01", overall_pick=3.0))
        results = repo.get_by_player_season(player_id, 2026)
        assert len(results) == 2

    def test_same_player_different_positions_coexist(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        repo = SqliteADPRepo(conn)
        repo.upsert(_make_adp(player_id, provider="fantasypros", positions="SP,DH", overall_pick=1.0, rank=1))
        repo.upsert(_make_adp(player_id, provider="fantasypros", positions="DH", overall_pick=2.0, rank=3))
        repo.upsert(_make_adp(player_id, provider="fantasypros", positions="SP", overall_pick=92.0, rank=95))
        results = repo.get_by_player_season(player_id, 2026)
        assert len(results) == 3
        by_pos = {r.positions: r.overall_pick for r in results}
        assert by_pos == {"SP,DH": 1.0, "DH": 2.0, "SP": 92.0}

    def test_get_empty_results(self, conn: sqlite3.Connection) -> None:
        seed_player(conn)
        repo = SqliteADPRepo(conn)
        assert repo.get_by_player_season(999, 2026) == []
        assert repo.get_by_season(2099) == []
