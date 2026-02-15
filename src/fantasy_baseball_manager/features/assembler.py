from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fantasy_baseball_manager.features.sql import generate_sql
from fantasy_baseball_manager.features.types import (
    DatasetHandle,
    DatasetSplits,
    DerivedTransformFeature,
    FeatureSet,
    Source,
    TransformFeature,
)


class SqliteDatasetAssembler:
    """Concrete DatasetAssembler backed by a single SQLite connection."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        statcast_path: Path | None = None,
    ) -> None:
        self._conn = conn
        self._statcast_path = statcast_path
        self._statcast_attached = False

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

    def _get_transform_features(self, feature_set: FeatureSet) -> list[TransformFeature]:
        return [f for f in feature_set.features if isinstance(f, TransformFeature)]

    def _add_transform_columns(
        self, table_name: str, transforms: list[TransformFeature] | list[DerivedTransformFeature]
    ) -> None:
        """ALTER TABLE to add REAL columns for each transform output."""
        for tf in transforms:
            for output in tf.outputs:
                self._conn.execute(f"ALTER TABLE [{table_name}] ADD COLUMN [{output}] REAL")

    def _attach_statcast(self) -> None:
        """Lazy ATTACH for the statcast DB."""
        if self._statcast_attached or self._statcast_path is None:
            return
        self._conn.execute(
            "ATTACH DATABASE ? AS [statcast]",
            (str(self._statcast_path),),
        )
        self._statcast_attached = True

    def _build_raw_query(
        self, tf: TransformFeature, table_name: str, *, player_type: str | None = None
    ) -> tuple[str, list[object]]:
        """Build a raw-row query for a transform feature."""
        if tf.source == Source.STATCAST:
            self._attach_statcast()
            sc_join_col = "pitcher_id" if player_type == "pitcher" else "batter_id"
            select_cols = ", ".join(f"sc.[{c}]" for c in tf.columns)
            sql = (
                f"SELECT d.player_id, "
                f"CAST(SUBSTR(sc.game_date, 1, 4) AS INTEGER) AS season, "
                f"{select_cols} "
                f"FROM [{table_name}] d "
                f"JOIN player p ON p.id = d.player_id "
                f"JOIN [statcast].statcast_pitch sc "
                f"ON sc.{sc_join_col} = p.mlbam_id "
                f"AND CAST(SUBSTR(sc.game_date, 1, 4) AS INTEGER) = d.season"
            )
            return sql, []

        # Non-statcast source: query from tables in the main DB
        source_tables = {
            Source.BATTING: "batting_stats",
            Source.PITCHING: "pitching_stats",
        }
        src_table = source_tables.get(tf.source, "batting_stats")
        select_cols = ", ".join(f"s.[{c}]" for c in tf.columns)
        sql = (
            f"SELECT d.player_id, s.season, {select_cols} "
            f"FROM [{table_name}] d "
            f"JOIN [{src_table}] s "
            f"ON s.player_id = d.player_id AND s.season = d.season"
        )
        return sql, []

    def _run_transforms(
        self, table_name: str, transforms: list[TransformFeature], *, player_type: str | None = None
    ) -> None:
        """Execute the transform pass: query raw rows, group, transform, update."""
        for tf in transforms:
            sql, params = self._build_raw_query(tf, table_name, player_type=player_type)
            cursor = self._conn.execute(sql, params)
            columns = [desc[0] for desc in cursor.description]
            all_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

            # Group by the transform's group_by key
            groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in all_rows:
                key = tuple(row[k] for k in tf.group_by)
                groups[key].append(row)

            # Apply transform and batch update
            output_cols = ", ".join(f"[{o}] = ?" for o in tf.outputs)
            where_parts = " AND ".join(f"[{k}] = ?" for k in tf.group_by)
            update_sql = f"UPDATE [{table_name}] SET {output_cols} WHERE {where_parts}"

            for group_key, group_rows in groups.items():
                result = tf.transform(group_rows)
                values = [result.get(o, 0.0) for o in tf.outputs]
                self._conn.execute(update_sql, [*values, *group_key])

            self._conn.commit()

    def _get_derived_transform_features(self, feature_set: FeatureSet) -> list[DerivedTransformFeature]:
        return [f for f in feature_set.features if isinstance(f, DerivedTransformFeature)]

    def _run_derived_transforms(self, table_name: str, transforms: list[DerivedTransformFeature]) -> None:
        """Execute the derived-transform pass: query from the dataset table itself."""
        for tf in transforms:
            input_cols = ", ".join(f"[{c}]" for c in tf.inputs)
            group_cols = ", ".join(f"[{k}]" for k in tf.group_by)
            sql = f"SELECT {group_cols}, {input_cols} FROM [{table_name}]"
            cursor = self._conn.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            all_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

            groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in all_rows:
                key = tuple(row[k] for k in tf.group_by)
                groups[key].append(row)

            output_cols = ", ".join(f"[{o}] = ?" for o in tf.outputs)
            where_parts = " AND ".join(f"[{k}] = ?" for k in tf.group_by)
            update_sql = f"UPDATE [{table_name}] SET {output_cols} WHERE {where_parts}"

            for group_key, group_rows in groups.items():
                result = tf.transform(group_rows)
                values = [result.get(o, 0.0) for o in tf.outputs]
                self._conn.execute(update_sql, [*values, *group_key])

            self._conn.commit()

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        # Pass 1: SQL materialization
        select_sql, params = generate_sql(feature_set)
        feature_set_id = self._ensure_feature_set(feature_set, select_sql)
        handle = self._create_dataset(feature_set_id, feature_set, select_sql, params)

        # Pass 2: Transform pass
        transforms = self._get_transform_features(feature_set)
        if transforms:
            self._add_transform_columns(handle.table_name, transforms)
            self._run_transforms(handle.table_name, transforms, player_type=feature_set.spine_filter.player_type)

        # Pass 3: Derived-transform pass
        derived = self._get_derived_transform_features(feature_set)
        if derived:
            self._add_transform_columns(handle.table_name, derived)
            self._run_derived_transforms(handle.table_name, derived)

        return handle

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        row = self._conn.execute(
            "SELECT id FROM feature_set WHERE name = ? AND version = ?",
            (feature_set.name, feature_set.version),
        ).fetchone()
        if row is None:
            return self.materialize(feature_set)

        feature_set_id: int = row[0]

        ds_row = self._conn.execute(
            "SELECT id, table_name, row_count, seasons FROM dataset WHERE feature_set_id = ? AND split IS NULL",
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
