from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from fantasy_baseball_manager.features.sql import generate_sql
from fantasy_baseball_manager.features.types import (
    DatasetHandle,
    DatasetSplits,
    FeatureSet,
)


class SqliteDatasetAssembler:
    """Concrete DatasetAssembler backed by a single SQLite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def _ensure_feature_set(self, feature_set: FeatureSet, source_query: str) -> int:
        """Insert feature_set row if missing, return feature_set_id."""
        row = self._conn.execute(
            "SELECT id FROM feature_set WHERE name = ? AND version = ?",
            (feature_set.name, feature_set.version),
        ).fetchone()
        if row is not None:
            return row[0]
        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO feature_set (name, version, source_query, created_at) VALUES (?, ?, ?, ?)",
            (feature_set.name, feature_set.version, source_query, now),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def _create_dataset(
        self, feature_set_id: int, feature_set: FeatureSet, select_sql: str, params: list[object]
    ) -> DatasetHandle:
        """Create a new dataset table and metadata row."""
        now = datetime.now(UTC).isoformat()
        seasons_json = json.dumps(list(feature_set.seasons))

        cursor = self._conn.execute(
            "INSERT INTO dataset (feature_set_id, name, split, table_name, row_count, seasons, created_at) "
            "VALUES (?, ?, NULL, NULL, 0, ?, ?)",
            (feature_set_id, feature_set.name, seasons_json, now),
        )
        dataset_id: int = cursor.lastrowid  # type: ignore[assignment]
        table_name = f"ds_{dataset_id}"

        self._conn.execute(f"CREATE TABLE [{table_name}] AS {select_sql}", params)

        row = self._conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()
        row_count: int = row[0]

        self._conn.execute(
            "UPDATE dataset SET table_name = ?, row_count = ? WHERE id = ?",
            (table_name, row_count, dataset_id),
        )
        self._conn.commit()

        return DatasetHandle(
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            table_name=table_name,
            row_count=row_count,
            seasons=feature_set.seasons,
        )

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        select_sql, params = generate_sql(feature_set)
        feature_set_id = self._ensure_feature_set(feature_set, select_sql)
        return self._create_dataset(feature_set_id, feature_set, select_sql, params)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        row = self._conn.execute(
            "SELECT id FROM feature_set WHERE name = ? AND version = ?",
            (feature_set.name, feature_set.version),
        ).fetchone()
        if row is None:
            return self.materialize(feature_set)

        feature_set_id: int = row[0]

        ds_row = self._conn.execute(
            "SELECT id, table_name, row_count, seasons FROM dataset " "WHERE feature_set_id = ? AND split IS NULL",
            (feature_set_id,),
        ).fetchone()
        if ds_row is not None:
            table_name: str = ds_row[1]
            table_exists = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if table_exists is not None:
                return DatasetHandle(
                    dataset_id=ds_row[0],
                    feature_set_id=feature_set_id,
                    table_name=table_name,
                    row_count=ds_row[2],
                    seasons=tuple(json.loads(ds_row[3])),
                )

        # Cache miss or stale — re-materialize
        select_sql, params = generate_sql(feature_set)
        return self._create_dataset(feature_set_id, feature_set, select_sql, params)

    def _create_split(
        self,
        handle: DatasetHandle,
        split_name: str,
        seasons: list[int],
    ) -> DatasetHandle:
        """Create a split table from the parent dataset table."""
        suffix = {"train": "train", "val": "val", "holdout": "holdout"}[split_name]
        split_table = f"{handle.table_name}_{suffix}"
        placeholders = ", ".join("?" for _ in seasons)
        self._conn.execute(
            f"CREATE TABLE [{split_table}] AS SELECT * FROM [{handle.table_name}] WHERE season IN ({placeholders})",
            seasons,
        )

        row = self._conn.execute(f"SELECT COUNT(*) FROM [{split_table}]").fetchone()
        row_count: int = row[0]

        now = datetime.now(UTC).isoformat()
        seasons_json = json.dumps(seasons)
        cursor = self._conn.execute(
            "INSERT INTO dataset (feature_set_id, name, split, table_name, row_count, seasons, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (handle.feature_set_id, split_table, split_name, split_table, row_count, seasons_json, now),
        )
        dataset_id: int = cursor.lastrowid  # type: ignore[assignment]

        return DatasetHandle(
            dataset_id=dataset_id,
            feature_set_id=handle.feature_set_id,
            table_name=split_table,
            row_count=row_count,
            seasons=tuple(seasons),
        )

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits:
        train_handle = self._create_split(handle, "train", list(train))
        val_handle = self._create_split(handle, "val", validation) if validation else None
        holdout_handle = self._create_split(handle, "holdout", holdout) if holdout else None
        self._conn.commit()
        return DatasetSplits(train=train_handle, validation=val_handle, holdout=holdout_handle)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        cursor = self._conn.execute(f"SELECT * FROM [{handle.table_name}]")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
