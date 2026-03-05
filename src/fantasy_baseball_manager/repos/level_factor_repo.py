from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import LevelFactor

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqliteLevelFactorRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, factor: LevelFactor) -> int:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                """INSERT INTO level_factor
                       (level, season, factor, k_factor, bb_factor, iso_factor, babip_factor, loaded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(level, season) DO UPDATE SET
                       factor=excluded.factor,
                       k_factor=excluded.k_factor,
                       bb_factor=excluded.bb_factor,
                       iso_factor=excluded.iso_factor,
                       babip_factor=excluded.babip_factor,
                       loaded_at=excluded.loaded_at""",
                (
                    factor.level,
                    factor.season,
                    factor.factor,
                    factor.k_factor,
                    factor.bb_factor,
                    factor.iso_factor,
                    factor.babip_factor,
                    factor.loaded_at,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_by_level_season(self, level: str, season: int) -> LevelFactor | None:
        with self._provider.connection() as conn:
            row = conn.execute(
                "SELECT * FROM level_factor WHERE level = ? AND season = ?",
                (level, season),
            ).fetchone()
            return self._row_to_factor(row) if row else None

    def get_by_season(self, season: int) -> list[LevelFactor]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM level_factor WHERE season = ?",
                (season,),
            ).fetchall()
            return [self._row_to_factor(row) for row in rows]

    @staticmethod
    def _row_to_factor(row: sqlite3.Row) -> LevelFactor:
        return LevelFactor(
            id=row["id"],
            level=row["level"],
            season=row["season"],
            factor=row["factor"],
            k_factor=row["k_factor"],
            bb_factor=row["bb_factor"],
            iso_factor=row["iso_factor"],
            babip_factor=row["babip_factor"],
            loaded_at=row["loaded_at"],
        )
