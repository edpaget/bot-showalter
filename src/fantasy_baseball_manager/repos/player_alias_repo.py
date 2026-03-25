from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PlayerAlias, PlayerType

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqlitePlayerAliasRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, alias: PlayerAlias) -> int:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO player_alias"
                "    (alias_name, player_id, player_type, source, active_from, active_to)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(alias_name, player_id, player_type) DO UPDATE SET"
                "    source=excluded.source,"
                "    active_from=excluded.active_from,"
                "    active_to=excluded.active_to",
                (
                    alias.alias_name,
                    alias.player_id,
                    alias.player_type or "",
                    alias.source,
                    alias.active_from,
                    alias.active_to,
                ),
            )
            conn.commit()
            assert cursor.lastrowid is not None  # noqa: S101
            return cursor.lastrowid

    def upsert_batch(self, aliases: list[PlayerAlias]) -> int:
        with self._provider.connection() as conn:
            count = 0
            for alias in aliases:
                conn.execute(
                    "INSERT INTO player_alias"
                    "    (alias_name, player_id, player_type, source, active_from, active_to)"
                    " VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(alias_name, player_id, player_type) DO UPDATE SET"
                    "    source=excluded.source,"
                    "    active_from=excluded.active_from,"
                    "    active_to=excluded.active_to",
                    (
                        alias.alias_name,
                        alias.player_id,
                        alias.player_type or "",
                        alias.source,
                        alias.active_from,
                        alias.active_to,
                    ),
                )
                count += 1
            conn.commit()
            return count

    def find_by_name(self, alias_name: str) -> list[PlayerAlias]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                self._select_sql() + " WHERE alias_name = ?",
                (alias_name,),
            ).fetchall()
            return [self._row_to_alias(row) for row in rows]

    def find_by_player(self, player_id: int) -> list[PlayerAlias]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                self._select_sql() + " WHERE player_id = ?",
                (player_id,),
            ).fetchall()
            return [self._row_to_alias(row) for row in rows]

    def delete_by_player(self, player_id: int) -> int:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM player_alias WHERE player_id = ?",
                (player_id,),
            )
            conn.commit()
            return cursor.rowcount

    @staticmethod
    def _select_sql() -> str:
        return "SELECT id, alias_name, player_id, player_type, source, active_from, active_to FROM player_alias"

    @staticmethod
    def _row_to_alias(row: sqlite3.Row) -> PlayerAlias:
        return PlayerAlias(
            id=row["id"],
            alias_name=row["alias_name"],
            player_id=row["player_id"],
            player_type=PlayerType(row["player_type"]) if row["player_type"] else None,
            source=row["source"],
            active_from=row["active_from"],
            active_to=row["active_to"],
        )
