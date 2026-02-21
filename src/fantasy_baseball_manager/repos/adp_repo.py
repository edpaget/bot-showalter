import sqlite3

from fantasy_baseball_manager.domain.adp import ADP


class SqliteADPRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, adp: ADP) -> int:
        as_of_db = adp.as_of if adp.as_of is not None else ""
        cursor = self._conn.execute(
            "INSERT INTO adp"
            "    (player_id, season, provider, overall_pick, rank,"
            "     positions, as_of, loaded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(player_id, season, provider, positions, as_of) DO UPDATE SET"
            "    overall_pick=excluded.overall_pick,"
            "    rank=excluded.rank,"
            "    loaded_at=excluded.loaded_at",
            (
                adp.player_id,
                adp.season,
                adp.provider,
                adp.overall_pick,
                adp.rank,
                adp.positions,
                as_of_db,
                adp.loaded_at,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_by_player_season(self, player_id: int, season: int) -> list[ADP]:
        rows = self._conn.execute(
            self._select_sql() + " WHERE player_id = ? AND season = ?",
            (player_id, season),
        ).fetchall()
        return [self._row_to_adp(row) for row in rows]

    def get_by_season(self, season: int, provider: str | None = None) -> list[ADP]:
        if provider is not None:
            rows = self._conn.execute(
                self._select_sql() + " WHERE season = ? AND provider = ?",
                (season, provider),
            ).fetchall()
        else:
            rows = self._conn.execute(
                self._select_sql() + " WHERE season = ?",
                (season,),
            ).fetchall()
        return [self._row_to_adp(row) for row in rows]

    def get_snapshots(self, season: int, provider: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT as_of FROM adp WHERE season = ? AND provider = ? AND as_of != '' ORDER BY as_of",
            (season, provider),
        ).fetchall()
        return [row["as_of"] for row in rows]

    def get_by_snapshot(self, season: int, provider: str, as_of: str) -> list[ADP]:
        rows = self._conn.execute(
            self._select_sql() + " WHERE season = ? AND provider = ? AND as_of = ?",
            (season, provider, as_of),
        ).fetchall()
        return [self._row_to_adp(row) for row in rows]

    @staticmethod
    def _select_sql() -> str:
        return "SELECT id, player_id, season, provider, overall_pick, rank, positions, as_of, loaded_at FROM adp"

    @staticmethod
    def _row_to_adp(row: sqlite3.Row) -> ADP:
        as_of_raw = row["as_of"]
        as_of = as_of_raw if as_of_raw != "" else None
        return ADP(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            provider=row["provider"],
            overall_pick=row["overall_pick"],
            rank=row["rank"],
            positions=row["positions"],
            as_of=as_of,
            loaded_at=row["loaded_at"],
        )
