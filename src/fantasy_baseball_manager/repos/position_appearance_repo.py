from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PositionAppearance

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqlitePositionAppearanceRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, appearance: PositionAppearance) -> int:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                """INSERT INTO position_appearance
                       (player_id, season, position, games, loaded_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(player_id, season, position) DO UPDATE SET
                       games=excluded.games,
                       loaded_at=excluded.loaded_at""",
                (
                    appearance.player_id,
                    appearance.season,
                    appearance.position,
                    appearance.games,
                    appearance.loaded_at,
                ),
            )
            return cursor.lastrowid

    def get_by_player(self, player_id: int) -> list[PositionAppearance]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM position_appearance WHERE player_id = ?",
                (player_id,),
            ).fetchall()
            return [self._row_to_appearance(row) for row in rows]

    def get_by_player_season(self, player_id: int, season: int) -> list[PositionAppearance]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM position_appearance WHERE player_id = ? AND season = ?",
                (player_id, season),
            ).fetchall()
            return [self._row_to_appearance(row) for row in rows]

    def get_by_season(self, season: int) -> list[PositionAppearance]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM position_appearance WHERE season = ?",
                (season,),
            ).fetchall()
            return [self._row_to_appearance(row) for row in rows]

    @staticmethod
    def _row_to_appearance(row: sqlite3.Row) -> PositionAppearance:
        return PositionAppearance(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            position=row["position"],
            games=row["games"],
            loaded_at=row["loaded_at"],
        )
