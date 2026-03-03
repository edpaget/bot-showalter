from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import FeatureCandidate

if TYPE_CHECKING:
    import sqlite3


class SqliteFeatureCandidateRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, candidate: FeatureCandidate) -> None:
        self._conn.execute(
            """INSERT INTO feature_candidate (name, expression, player_type, min_pa, min_ip, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                   expression = excluded.expression,
                   player_type = excluded.player_type,
                   min_pa = excluded.min_pa,
                   min_ip = excluded.min_ip,
                   created_at = excluded.created_at""",
            (
                candidate.name,
                candidate.expression,
                candidate.player_type,
                candidate.min_pa,
                candidate.min_ip,
                candidate.created_at,
            ),
        )
        self._conn.commit()

    def get_by_name(self, name: str) -> FeatureCandidate | None:
        row = self._conn.execute(
            "SELECT name, expression, player_type, min_pa, min_ip, created_at FROM feature_candidate WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return FeatureCandidate(
            name=row[0],
            expression=row[1],
            player_type=row[2],
            min_pa=row[3],
            min_ip=row[4],
            created_at=row[5],
        )

    def list_all(self) -> list[FeatureCandidate]:
        rows = self._conn.execute(
            "SELECT name, expression, player_type, min_pa, min_ip, created_at FROM feature_candidate ORDER BY name"
        ).fetchall()
        return [
            FeatureCandidate(
                name=row[0],
                expression=row[1],
                player_type=row[2],
                min_pa=row[3],
                min_ip=row[4],
                created_at=row[5],
            )
            for row in rows
        ]

    def delete(self, name: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM feature_candidate WHERE name = ?",
            (name,),
        )
        self._conn.commit()
        return cursor.rowcount > 0
