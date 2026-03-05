from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import FeatureCandidate

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqliteFeatureCandidateRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def save(self, candidate: FeatureCandidate) -> None:
        with self._provider.connection() as conn:
            conn.execute(
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
            conn.commit()

    def get_by_name(self, name: str) -> FeatureCandidate | None:
        with self._provider.connection() as conn:
            row = conn.execute(
                "SELECT name, expression, player_type, min_pa, min_ip, created_at"
                " FROM feature_candidate WHERE name = ?",
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
        with self._provider.connection() as conn:
            rows = conn.execute(
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
        with self._provider.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM feature_candidate WHERE name = ?",
                (name,),
            )
            conn.commit()
            return cursor.rowcount > 0
