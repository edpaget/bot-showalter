from typing import TYPE_CHECKING

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.yahoo_draft_pick import YahooDraftPick
from fantasy_baseball_manager.repos.yahoo_draft_repo import SqliteYahooDraftRepo

if TYPE_CHECKING:
    import sqlite3


def _make_pick(**overrides: object) -> YahooDraftPick:
    defaults: dict[str, object] = {
        "league_key": "449.l.12345",
        "season": 2026,
        "round": 1,
        "pick": 1,
        "team_key": "449.l.12345.t.1",
        "yahoo_player_key": "449.p.1234",
        "player_id": 545361,
        "player_name": "Mike Trout",
        "position": "OF",
    }
    defaults.update(overrides)
    return YahooDraftPick(**defaults)  # type: ignore[arg-type]


class TestSqliteYahooDraftRepo:
    def test_upsert_and_get(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooDraftRepo(SingleConnectionProvider(conn))
        pick = _make_pick()
        repo.upsert(pick)

        results = repo.get_by_league_season("449.l.12345", 2026)
        assert len(results) == 1
        assert results[0].league_key == "449.l.12345"
        assert results[0].season == 2026
        assert results[0].round == 1
        assert results[0].pick == 1
        assert results[0].team_key == "449.l.12345.t.1"
        assert results[0].yahoo_player_key == "449.p.1234"
        assert results[0].player_id == 545361
        assert results[0].player_name == "Mike Trout"
        assert results[0].position == "OF"
        assert results[0].cost is None
        assert results[0].id is not None

    def test_upsert_on_conflict_updates(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooDraftRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_pick(player_name="Mike Trout"))
        repo.upsert(_make_pick(player_name="Mike Trout Updated", player_id=999))

        results = repo.get_by_league_season("449.l.12345", 2026)
        assert len(results) == 1
        assert results[0].player_name == "Mike Trout Updated"
        assert results[0].player_id == 999

    def test_get_by_league_season_ordered_by_round_pick(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooDraftRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_pick(round=2, pick=3, player_name="Third"))
        repo.upsert(_make_pick(round=1, pick=1, player_name="First"))
        repo.upsert(_make_pick(round=2, pick=1, player_name="Second"))

        results = repo.get_by_league_season("449.l.12345", 2026)
        assert len(results) == 3
        assert results[0].player_name == "First"
        assert results[1].player_name == "Second"
        assert results[2].player_name == "Third"

    def test_get_by_league_season_empty(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooDraftRepo(SingleConnectionProvider(conn))
        results = repo.get_by_league_season("nonexistent", 2026)
        assert results == []

    def test_get_pick_count(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooDraftRepo(SingleConnectionProvider(conn))
        assert repo.get_pick_count("449.l.12345", 2026) == 0

        repo.upsert(_make_pick(round=1, pick=1))
        repo.upsert(_make_pick(round=1, pick=2, yahoo_player_key="449.p.5678"))
        assert repo.get_pick_count("449.l.12345", 2026) == 2

    def test_round_trip_with_cost(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooDraftRepo(SingleConnectionProvider(conn))
        pick = _make_pick(cost=45)
        repo.upsert(pick)

        results = repo.get_by_league_season("449.l.12345", 2026)
        assert len(results) == 1
        assert results[0].cost == 45

    def test_round_trip_nullable_player_id(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooDraftRepo(SingleConnectionProvider(conn))
        pick = _make_pick(player_id=None, player_name="Unknown Player")
        repo.upsert(pick)

        results = repo.get_by_league_season("449.l.12345", 2026)
        assert len(results) == 1
        assert results[0].player_id is None
        assert results[0].player_name == "Unknown Player"

    def test_different_seasons_separated(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooDraftRepo(SingleConnectionProvider(conn))
        repo.upsert(_make_pick(season=2025, player_name="2025 Pick"))
        repo.upsert(_make_pick(season=2026, player_name="2026 Pick"))

        results_2025 = repo.get_by_league_season("449.l.12345", 2025)
        results_2026 = repo.get_by_league_season("449.l.12345", 2026)
        assert len(results_2025) == 1
        assert results_2025[0].player_name == "2025 Pick"
        assert len(results_2026) == 1
        assert results_2026[0].player_name == "2026 Pick"
