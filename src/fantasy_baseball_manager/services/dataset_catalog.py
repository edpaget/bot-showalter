import json
import logging
import sqlite3
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: int
    feature_set_id: int
    feature_set_name: str
    feature_set_version: str
    split: str | None
    table_name: str
    row_count: int
    seasons: list[int]
    created_at: str


class DatasetCatalogService:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def _rows_to_infos(self, rows: list[tuple[object, ...]]) -> list[DatasetInfo]:
        return [
            DatasetInfo(
                dataset_id=row[0],  # type: ignore[arg-type]
                feature_set_id=row[1],  # type: ignore[arg-type]
                feature_set_name=row[2],  # type: ignore[arg-type]
                feature_set_version=row[3],  # type: ignore[arg-type]
                split=row[4],  # type: ignore[arg-type]
                table_name=row[5],  # type: ignore[arg-type]
                row_count=row[6],  # type: ignore[arg-type]
                seasons=json.loads(row[7]) if row[7] else [],  # type: ignore[arg-type]
                created_at=row[8],  # type: ignore[arg-type]
            )
            for row in rows
        ]

    _SELECT = (
        "SELECT d.id, d.feature_set_id, fs.name, fs.version, d.split, "
        "d.table_name, d.row_count, d.seasons, d.created_at "
        "FROM dataset d JOIN feature_set fs ON fs.id = d.feature_set_id"
    )

    def list_all(self) -> list[DatasetInfo]:
        rows = self._conn.execute(f"{self._SELECT} ORDER BY fs.name, d.id").fetchall()
        return self._rows_to_infos(rows)

    def list_by_feature_set_name(self, name: str) -> list[DatasetInfo]:
        rows = self._conn.execute(
            f"{self._SELECT} WHERE fs.name = ? ORDER BY d.id",
            (name,),
        ).fetchall()
        return self._rows_to_infos(rows)

    def _drop_datasets(self, where_clause: str, params: tuple[object, ...]) -> int:
        """Drop dataset tables and metadata matching the WHERE clause on feature_set."""
        # Find all datasets for matching feature sets
        rows = self._conn.execute(
            f"SELECT d.id, d.table_name, d.feature_set_id FROM dataset d "
            f"JOIN feature_set fs ON fs.id = d.feature_set_id WHERE {where_clause}",
            params,
        ).fetchall()

        if not rows:
            logger.debug("No datasets matched for drop")
            return 0

        logger.info("Dropping %d dataset(s)", len(rows))
        feature_set_ids: set[int] = set()
        for row in rows:
            table_name: str = row[1]
            feature_set_ids.add(row[2])
            # DROP TABLE IF EXISTS — skip if already gone
            self._conn.execute(f"DROP TABLE IF EXISTS [{table_name}]")

        # Delete dataset metadata
        dataset_ids = [row[0] for row in rows]
        placeholders = ", ".join("?" for _ in dataset_ids)
        self._conn.execute(f"DELETE FROM dataset WHERE id IN ({placeholders})", dataset_ids)

        # Delete orphaned feature_set rows (no remaining datasets)
        for fs_id in feature_set_ids:
            remaining = self._conn.execute(
                "SELECT COUNT(*) FROM dataset WHERE feature_set_id = ?", (fs_id,)
            ).fetchone()[0]
            if remaining == 0:
                self._conn.execute("DELETE FROM feature_set WHERE id = ?", (fs_id,))

        return len(rows)

    def drop_by_feature_set_name(self, name: str) -> int:
        return self._drop_datasets("fs.name = ?", (name,))

    def drop_by_name_prefix(self, prefix: str) -> int:
        return self._drop_datasets("fs.name LIKE ?", (f"{prefix}%",))

    def drop_all(self) -> int:
        return self._drop_datasets("1=1", ())
