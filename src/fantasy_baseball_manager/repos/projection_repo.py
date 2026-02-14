import json
import sqlite3

from fantasy_baseball_manager.domain.projection import Projection


class SqliteProjectionRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, projection: Projection) -> int:
        stat_json_str = json.dumps(projection.stat_json)
        cursor = self._conn.execute(
            """INSERT INTO projection
                   (player_id, season, system, version, player_type, stat_json, loaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(player_id, season, system, version, player_type) DO UPDATE SET
                   stat_json=excluded.stat_json, loaded_at=excluded.loaded_at""",
            (
                projection.player_id,
                projection.season,
                projection.system,
                projection.version,
                projection.player_type,
                stat_json_str,
                projection.loaded_at,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_player_season(self, player_id: int, season: int, system: str | None = None) -> list[Projection]:
        if system is not None:
            rows = self._conn.execute(
                "SELECT * FROM projection WHERE player_id = ? AND season = ? AND system = ?",
                (player_id, season, system),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM projection WHERE player_id = ? AND season = ?",
                (player_id, season),
            ).fetchall()
        return [self._row_to_projection(row) for row in rows]

    def get_by_season(self, season: int, system: str | None = None) -> list[Projection]:
        if system is not None:
            rows = self._conn.execute(
                "SELECT * FROM projection WHERE season = ? AND system = ?",
                (season, system),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM projection WHERE season = ?",
                (season,),
            ).fetchall()
        return [self._row_to_projection(row) for row in rows]

    def get_by_system_version(self, system: str, version: str) -> list[Projection]:
        rows = self._conn.execute(
            "SELECT * FROM projection WHERE system = ? AND version = ?",
            (system, version),
        ).fetchall()
        return [self._row_to_projection(row) for row in rows]

    @staticmethod
    def _row_to_projection(row: tuple) -> Projection:
        return Projection(
            id=row[0],
            player_id=row[1],
            season=row[2],
            system=row[3],
            version=row[4],
            player_type=row[5],
            stat_json=json.loads(row[6]),
            loaded_at=row[7],
        )
