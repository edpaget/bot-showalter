import sqlite3

from fantasy_baseball_manager.domain.load_log import LoadLog


class SqliteLoadLogRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, log: LoadLog) -> int:
        cursor = self._conn.execute(
            """INSERT INTO load_log
                   (source_type, source_detail, target_table, rows_loaded,
                    started_at, finished_at, status, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                log.source_type,
                log.source_detail,
                log.target_table,
                log.rows_loaded,
                log.started_at,
                log.finished_at,
                log.status,
                log.error_message,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get_recent(self, limit: int = 20) -> list[LoadLog]:
        rows = self._conn.execute(
            "SELECT * FROM load_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_log(row) for row in rows]

    def get_by_target_table(self, target_table: str) -> list[LoadLog]:
        rows = self._conn.execute(
            "SELECT * FROM load_log WHERE target_table = ?",
            (target_table,),
        ).fetchall()
        return [self._row_to_log(row) for row in rows]

    @staticmethod
    def _row_to_log(row: sqlite3.Row) -> LoadLog:
        return LoadLog(
            id=row["id"],
            source_type=row["source_type"],
            source_detail=row["source_detail"],
            target_table=row["target_table"],
            rows_loaded=row["rows_loaded"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            status=row["status"],
            error_message=row["error_message"],
        )
