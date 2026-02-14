import sqlite3

from fantasy_baseball_manager.domain.projection import Projection

_STAT_COLUMNS: tuple[str, ...] = (
    "pa",
    "ab",
    "h",
    "doubles",
    "triples",
    "hr",
    "rbi",
    "r",
    "sb",
    "cs",
    "bb",
    "so",
    "hbp",
    "sf",
    "sh",
    "gdp",
    "ibb",
    "avg",
    "obp",
    "slg",
    "ops",
    "woba",
    "wrc_plus",
    "war",
    "w",
    "l",
    "era",
    "g",
    "gs",
    "sv",
    "hld",
    "ip",
    "er",
    "whip",
    "k_per_9",
    "bb_per_9",
    "fip",
    "xfip",
)


class SqliteProjectionRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, projection: Projection) -> int:
        stat_values = [projection.stat_json.get(col) for col in _STAT_COLUMNS]
        stat_placeholders = ", ".join("?" for _ in _STAT_COLUMNS)
        stat_col_list = ", ".join(_STAT_COLUMNS)
        stat_update = ", ".join(f"{col}=excluded.{col}" for col in _STAT_COLUMNS)
        cursor = self._conn.execute(
            f"INSERT INTO projection"
            f"    (player_id, season, system, version, player_type, {stat_col_list}, loaded_at, source_type)"
            f" VALUES (?, ?, ?, ?, ?, {stat_placeholders}, ?, ?)"
            f" ON CONFLICT(player_id, season, system, version, player_type) DO UPDATE SET"
            f"    {stat_update}, loaded_at=excluded.loaded_at, source_type=excluded.source_type",
            (
                projection.player_id,
                projection.season,
                projection.system,
                projection.version,
                projection.player_type,
                *stat_values,
                projection.loaded_at,
                projection.source_type,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_player_season(self, player_id: int, season: int, system: str | None = None) -> list[Projection]:
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
        return [self._row_to_projection(row) for row in rows]

    def get_by_season(self, season: int, system: str | None = None) -> list[Projection]:
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
        return [self._row_to_projection(row) for row in rows]

    def get_by_system_version(self, system: str, version: str) -> list[Projection]:
        rows = self._conn.execute(
            self._select_sql() + " WHERE system = ? AND version = ?",
            (system, version),
        ).fetchall()
        return [self._row_to_projection(row) for row in rows]

    @staticmethod
    def _select_sql() -> str:
        stat_col_list = ", ".join(_STAT_COLUMNS)
        return f"SELECT id, player_id, season, system, version, player_type, {stat_col_list}, loaded_at, source_type FROM projection"

    @staticmethod
    def _row_to_projection(row: sqlite3.Row) -> Projection:
        stat_json = {col: row[col] for col in _STAT_COLUMNS if row[col] is not None}
        return Projection(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            system=row["system"],
            version=row["version"],
            player_type=row["player_type"],
            stat_json=stat_json,
            loaded_at=row["loaded_at"],
            source_type=row["source_type"],
        )
