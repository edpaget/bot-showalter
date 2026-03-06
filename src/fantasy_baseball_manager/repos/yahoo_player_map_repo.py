from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import YahooPlayerMap

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqliteYahooPlayerMapRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, mapping: YahooPlayerMap) -> int:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO yahoo_player_map"
                "    (yahoo_player_key, player_id, player_type, yahoo_name, yahoo_team, yahoo_positions)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(yahoo_player_key) DO UPDATE SET"
                "    player_id=excluded.player_id,"
                "    player_type=excluded.player_type,"
                "    yahoo_name=excluded.yahoo_name,"
                "    yahoo_team=excluded.yahoo_team,"
                "    yahoo_positions=excluded.yahoo_positions",
                (
                    mapping.yahoo_player_key,
                    mapping.player_id,
                    mapping.player_type,
                    mapping.yahoo_name,
                    mapping.yahoo_team,
                    mapping.yahoo_positions,
                ),
            )
            return cursor.lastrowid

    def get_by_yahoo_key(self, yahoo_player_key: str) -> YahooPlayerMap | None:
        with self._provider.connection() as conn:
            row = conn.execute(
                self._select_sql() + " WHERE yahoo_player_key = ?",
                (yahoo_player_key,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_mapping(row)

    def get_by_player_id(self, player_id: int) -> list[YahooPlayerMap]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                self._select_sql() + " WHERE player_id = ?",
                (player_id,),
            ).fetchall()
            return [self._row_to_mapping(row) for row in rows]

    def get_all(self) -> list[YahooPlayerMap]:
        with self._provider.connection() as conn:
            rows = conn.execute(self._select_sql()).fetchall()
            return [self._row_to_mapping(row) for row in rows]

    @staticmethod
    def _select_sql() -> str:
        return (
            "SELECT id, yahoo_player_key, player_id, player_type,"
            " yahoo_name, yahoo_team, yahoo_positions"
            " FROM yahoo_player_map"
        )

    @staticmethod
    def _row_to_mapping(row: sqlite3.Row) -> YahooPlayerMap:
        return YahooPlayerMap(
            id=row["id"],
            yahoo_player_key=row["yahoo_player_key"],
            player_id=row["player_id"],
            player_type=row["player_type"],
            yahoo_name=row["yahoo_name"],
            yahoo_team=row["yahoo_team"],
            yahoo_positions=row["yahoo_positions"],
        )
