import sqlite3

from fantasy_baseball_manager.domain.sprint_speed import SprintSpeed


class SqliteSprintSpeedRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, record: SprintSpeed) -> int:
        cursor = self._conn.execute(
            """INSERT INTO sprint_speed
                   (mlbam_id, season, sprint_speed, hp_to_1b,
                    bolts, competitive_runs, loaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(mlbam_id, season) DO UPDATE SET
                   sprint_speed=excluded.sprint_speed,
                   hp_to_1b=excluded.hp_to_1b,
                   bolts=excluded.bolts,
                   competitive_runs=excluded.competitive_runs,
                   loaded_at=excluded.loaded_at""",
            (
                record.mlbam_id,
                record.season,
                record.sprint_speed,
                record.hp_to_1b,
                record.bolts,
                record.competitive_runs,
                record.loaded_at,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM sprint_speed").fetchone()
        return row[0]
