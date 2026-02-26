from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.yahoo_player import YahooPlayerMap
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.yahoo_player_map_repo import SqliteYahooPlayerMapRepo

if TYPE_CHECKING:
    import sqlite3


def _seed_player(conn: sqlite3.Connection, player_id: int, name: str = "Test Player") -> int:
    parts = name.split(" ", 1)
    first, last = (parts[0], parts[1]) if len(parts) == 2 else (name, "Unknown")
    repo = SqlitePlayerRepo(conn)
    return repo.upsert(Player(name_first=first, name_last=last, mlbam_id=player_id + 100000))


def _make_mapping(**overrides: object) -> YahooPlayerMap:
    defaults: dict[str, object] = {
        "yahoo_player_key": "449.p.12345",
        "player_id": 42,
        "player_type": "batter",
        "yahoo_name": "Mike Trout",
        "yahoo_team": "LAA",
        "yahoo_positions": "CF,LF",
    }
    defaults.update(overrides)
    return YahooPlayerMap(**defaults)  # type: ignore[arg-type]


class TestSqliteYahooPlayerMapRepo:
    def test_upsert_and_get_by_yahoo_key(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn, 42, "Mike Trout")
        repo = SqliteYahooPlayerMapRepo(conn)
        repo.upsert(_make_mapping(player_id=player_id))
        result = repo.get_by_yahoo_key("449.p.12345")
        assert result is not None
        assert result.yahoo_player_key == "449.p.12345"
        assert result.player_id == player_id
        assert result.player_type == "batter"
        assert result.yahoo_name == "Mike Trout"
        assert result.yahoo_team == "LAA"
        assert result.yahoo_positions == "CF,LF"
        assert result.id is not None

    def test_get_by_yahoo_key_returns_none_for_missing(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooPlayerMapRepo(conn)
        assert repo.get_by_yahoo_key("nonexistent") is None

    def test_get_by_player_id(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn, 42, "Mike Trout")
        repo = SqliteYahooPlayerMapRepo(conn)
        repo.upsert(_make_mapping(player_id=player_id))
        results = repo.get_by_player_id(player_id)
        assert len(results) == 1
        assert results[0].yahoo_player_key == "449.p.12345"

    def test_get_by_player_id_two_way_player(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn, 100, "Shohei Ohtani")
        repo = SqliteYahooPlayerMapRepo(conn)
        repo.upsert(
            _make_mapping(
                yahoo_player_key="449.p.11111",
                player_id=player_id,
                player_type="batter",
                yahoo_name="Shohei Ohtani",
                yahoo_team="LAD",
                yahoo_positions="DH",
            )
        )
        repo.upsert(
            _make_mapping(
                yahoo_player_key="449.p.22222",
                player_id=player_id,
                player_type="pitcher",
                yahoo_name="Shohei Ohtani",
                yahoo_team="LAD",
                yahoo_positions="SP",
            )
        )
        results = repo.get_by_player_id(player_id)
        assert len(results) == 2
        types = {r.player_type for r in results}
        assert types == {"batter", "pitcher"}

    def test_get_by_player_id_returns_empty_for_missing(self, conn: sqlite3.Connection) -> None:
        repo = SqliteYahooPlayerMapRepo(conn)
        assert repo.get_by_player_id(999) == []

    def test_upsert_updates_existing(self, conn: sqlite3.Connection) -> None:
        player_id = _seed_player(conn, 42, "Mike Trout")
        repo = SqliteYahooPlayerMapRepo(conn)
        repo.upsert(_make_mapping(player_id=player_id, yahoo_name="Old Name"))
        repo.upsert(_make_mapping(player_id=player_id, yahoo_name="New Name"))
        result = repo.get_by_yahoo_key("449.p.12345")
        assert result is not None
        assert result.yahoo_name == "New Name"

    def test_get_all(self, conn: sqlite3.Connection) -> None:
        pid1 = _seed_player(conn, 1, "Player One")
        pid2 = _seed_player(conn, 2, "Player Two")
        repo = SqliteYahooPlayerMapRepo(conn)
        repo.upsert(_make_mapping(yahoo_player_key="449.p.1", player_id=pid1))
        repo.upsert(_make_mapping(yahoo_player_key="449.p.2", player_id=pid2))
        results = repo.get_all()
        assert len(results) == 2
