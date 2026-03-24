import dataclasses
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PlayerType, Projection, StatDistribution

if TYPE_CHECKING:
    import sqlite3

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider

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
    "iso",
    "babip",
    "hr_per_9",
    "k_pct",
    "bb_pct",
)


class SqliteProjectionRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def upsert(self, projection: Projection) -> int:
        with self._provider.connection() as conn:
            stat_values = [projection.stat_json.get(col) for col in _STAT_COLUMNS]
            stat_placeholders = ", ".join("?" for _ in _STAT_COLUMNS)
            stat_col_list = ", ".join(_STAT_COLUMNS)
            stat_update = ", ".join(f"{col}=excluded.{col}" for col in _STAT_COLUMNS)
            conn.execute(
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
            # lastrowid is unreliable for ON CONFLICT DO UPDATE; query the actual ID.
            row = conn.execute(
                "SELECT id FROM projection"
                " WHERE player_id = ? AND season = ? AND system = ? AND version = ? AND player_type = ?",
                (
                    projection.player_id,
                    projection.season,
                    projection.system,
                    projection.version,
                    projection.player_type,
                ),
            ).fetchone()
            return row[0]

    def get_by_player_season(
        self, player_id: int, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        with self._provider.connection() as conn:
            if system is not None:
                rows = conn.execute(
                    self._select_sql() + " WHERE player_id = ? AND season = ? AND system = ?",
                    (player_id, season, system),
                ).fetchall()
            else:
                rows = conn.execute(
                    self._select_sql() + " WHERE player_id = ? AND season = ?",
                    (player_id, season),
                ).fetchall()
            projections = [self._row_to_projection(row) for row in rows]
            if include_distributions:
                projections = self._attach_distributions(projections)
            return projections

    def get_by_season(
        self, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Projection]:
        with self._provider.connection() as conn:
            if system is not None:
                rows = conn.execute(
                    self._select_sql() + " WHERE season = ? AND system = ?",
                    (season, system),
                ).fetchall()
            else:
                rows = conn.execute(
                    self._select_sql() + " WHERE season = ?",
                    (season,),
                ).fetchall()
            projections = [self._row_to_projection(row) for row in rows]
            if include_distributions:
                projections = self._attach_distributions(projections)
            return projections

    def get_by_system_version(self, system: str, version: str) -> list[Projection]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                self._select_sql() + " WHERE system = ? AND version = ?",
                (system, version),
            ).fetchall()
            return [self._row_to_projection(row) for row in rows]

    def delete_by_system_version(self, system: str, version: str) -> int:
        with self._provider.connection() as conn:
            conn.execute(
                "DELETE FROM projection_distribution WHERE projection_id IN"
                " (SELECT id FROM projection WHERE system = ? AND version = ?)",
                (system, version),
            )
            cursor = conn.execute(
                "DELETE FROM projection WHERE system = ? AND version = ?",
                (system, version),
            )
            return cursor.rowcount

    def upsert_distributions(self, projection_id: int, distributions: list[StatDistribution]) -> None:
        with self._provider.connection() as conn:
            conn.executemany(
                "INSERT INTO projection_distribution"
                " (projection_id, stat, p10, p25, p50, p75, p90, mean, std, family)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(projection_id, stat) DO UPDATE SET"
                " p10=excluded.p10, p25=excluded.p25, p50=excluded.p50,"
                " p75=excluded.p75, p90=excluded.p90, mean=excluded.mean,"
                " std=excluded.std, family=excluded.family",
                [
                    (projection_id, d.stat, d.p10, d.p25, d.p50, d.p75, d.p90, d.mean, d.std, d.family)
                    for d in distributions
                ],
            )

    def get_distributions(self, projection_id: int) -> list[StatDistribution]:
        with self._provider.connection() as conn:
            rows = conn.execute(
                "SELECT stat, p10, p25, p50, p75, p90, mean, std, family"
                " FROM projection_distribution WHERE projection_id = ?",
                (projection_id,),
            ).fetchall()
            return [
                StatDistribution(
                    stat=row["stat"],
                    p10=row["p10"],
                    p25=row["p25"],
                    p50=row["p50"],
                    p75=row["p75"],
                    p90=row["p90"],
                    mean=row["mean"],
                    std=row["std"],
                    family=row["family"],
                )
                for row in rows
            ]

    def _attach_distributions(self, projections: list[Projection]) -> list[Projection]:
        with self._provider.connection() as conn:
            if not projections:
                return projections
            proj_ids = [p.id for p in projections if p.id is not None]
            if not proj_ids:
                return projections
            placeholders = ", ".join("?" for _ in proj_ids)
            rows = conn.execute(
                f"SELECT projection_id, stat, p10, p25, p50, p75, p90, mean, std, family"
                f" FROM projection_distribution WHERE projection_id IN ({placeholders})",
                proj_ids,
            ).fetchall()
            dist_map: dict[int, dict[str, StatDistribution]] = {}
            for row in rows:
                pid = row["projection_id"]
                dist = StatDistribution(
                    stat=row["stat"],
                    p10=row["p10"],
                    p25=row["p25"],
                    p50=row["p50"],
                    p75=row["p75"],
                    p90=row["p90"],
                    mean=row["mean"],
                    std=row["std"],
                    family=row["family"],
                )
                dist_map.setdefault(pid, {})[dist.stat] = dist
            return [
                dataclasses.replace(p, distributions=dist_map.get(p.id, {})) if p.id is not None else p
                for p in projections
            ]

    @staticmethod
    def _select_sql() -> str:
        stat_col_list = ", ".join(_STAT_COLUMNS)
        return (
            f"SELECT id, player_id, season, system, version, player_type,"
            f" {stat_col_list}, loaded_at, source_type FROM projection"
        )

    @staticmethod
    def _row_to_projection(row: sqlite3.Row) -> Projection:
        stat_json = {col: row[col] for col in _STAT_COLUMNS if row[col] is not None}
        return Projection(
            id=row["id"],
            player_id=row["player_id"],
            season=row["season"],
            system=row["system"],
            version=row["version"],
            player_type=PlayerType(row["player_type"]),
            stat_json=stat_json,
            loaded_at=row["loaded_at"],
            source_type=row["source_type"],
        )
