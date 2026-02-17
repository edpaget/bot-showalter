import json
import sqlite3

from fantasy_baseball_manager.domain.valuation import Valuation


class SqliteValuationRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, valuation: Valuation) -> int:
        cursor = self._conn.execute(
            "INSERT INTO valuation"
            "    (player_id, season, system, version, projection_system,"
            "     projection_version, player_type, position, value, rank,"
            "     category_scores_json, loaded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(player_id, season, system, version, player_type) DO UPDATE SET"
            "    projection_system=excluded.projection_system,"
            "    projection_version=excluded.projection_version,"
            "    position=excluded.position,"
            "    value=excluded.value,"
            "    rank=excluded.rank,"
            "    category_scores_json=excluded.category_scores_json,"
            "    loaded_at=excluded.loaded_at",
            (
                valuation.player_id,
                valuation.season,
                valuation.system,
                valuation.version,
                valuation.projection_system,
                valuation.projection_version,
                valuation.player_type,
                valuation.position,
                valuation.value,
                valuation.rank,
                json.dumps(valuation.category_scores),
                valuation.loaded_at,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_player_season(self, player_id: int, season: int, system: str | None = None) -> list[Valuation]:
        if system is not None:
            rows = self._conn.execute(
                self._select_sql() + " WHERE player_id = ? AND season = ? AND system = ?",
                (player_id, season, system),
            ).fetchall()
        else:
            rows = self._conn.execute(
                self._select_sql() + " WHERE player_id = ? AND season = ?",
                (player_id, season),
            ).fetchall()
        return [self._row_to_valuation(row) for row in rows]

    def get_by_season(self, season: int, system: str | None = None) -> list[Valuation]:
        if system is not None:
            rows = self._conn.execute(
                self._select_sql() + " WHERE season = ? AND system = ?",
                (season, system),
            ).fetchall()
        else:
            rows = self._conn.execute(
                self._select_sql() + " WHERE season = ?",
                (season,),
            ).fetchall()
        return [self._row_to_valuation(row) for row in rows]

    @staticmethod
    def _select_sql() -> str:
        return (
            "SELECT id, player_id, season, system, version,"
            " projection_system, projection_version, player_type,"
            " position, value, rank, category_scores_json, loaded_at"
            " FROM valuation"
        )

    @staticmethod
    def _row_to_valuation(row: sqlite3.Row) -> Valuation:
        return Valuation(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            system=row["system"],
            version=row["version"],
            projection_system=row["projection_system"],
            projection_version=row["projection_version"],
            player_type=row["player_type"],
            position=row["position"],
            value=row["value"],
            rank=row["rank"],
            category_scores=json.loads(row["category_scores_json"]),
            loaded_at=row["loaded_at"],
        )
