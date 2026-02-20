import sqlite3

from tests.helpers import seed_player


class TestSeedPlayerAutoId:
    def test_returns_auto_id(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn)
        assert isinstance(player_id, int)
        assert player_id > 0

    def test_explicit_player_id(self, conn: sqlite3.Connection) -> None:
        result = seed_player(conn, player_id=42)
        assert result == 42
        row = conn.execute("SELECT id, name_first, name_last FROM player WHERE id = 42").fetchone()
        assert row is not None
        assert row["name_first"] == "Test"
        assert row["name_last"] == "Player"

    def test_custom_fields(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, name_first="Mike", name_last="Trout", mlbam_id=545361)
        row = conn.execute("SELECT name_first, name_last, mlbam_id FROM player WHERE id = ?", (player_id,)).fetchone()
        assert row["name_first"] == "Mike"
        assert row["name_last"] == "Trout"
        assert row["mlbam_id"] == 545361

    def test_multiple_players_distinct_ids(self, conn: sqlite3.Connection) -> None:
        id1 = seed_player(conn)
        id2 = seed_player(conn, mlbam_id=660271)
        assert id1 != id2
