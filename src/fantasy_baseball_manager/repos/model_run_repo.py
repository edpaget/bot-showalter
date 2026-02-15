from __future__ import annotations

import builtins
import json
import sqlite3

from fantasy_baseball_manager.domain.model_run import ModelRunRecord


class SqliteModelRunRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, record: ModelRunRecord) -> int:
        cursor = self._conn.execute(
            """INSERT INTO model_run
                   (system, version, operation, train_dataset_id, validation_dataset_id,
                    holdout_dataset_id, config_json, metrics_json, artifact_type,
                    artifact_path, git_commit, tags_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(system, version, operation) DO UPDATE SET
                   train_dataset_id=excluded.train_dataset_id,
                   validation_dataset_id=excluded.validation_dataset_id,
                   holdout_dataset_id=excluded.holdout_dataset_id,
                   config_json=excluded.config_json,
                   metrics_json=excluded.metrics_json,
                   artifact_type=excluded.artifact_type,
                   artifact_path=excluded.artifact_path,
                   git_commit=excluded.git_commit,
                   tags_json=excluded.tags_json,
                   created_at=excluded.created_at""",
            (
                record.system,
                record.version,
                record.operation,
                record.train_dataset_id,
                record.validation_dataset_id,
                record.holdout_dataset_id,
                json.dumps(record.config_json),
                json.dumps(record.metrics_json) if record.metrics_json is not None else None,
                record.artifact_type,
                record.artifact_path,
                record.git_commit,
                json.dumps(record.tags_json) if record.tags_json is not None else None,
                record.created_at,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get(self, system: str, version: str, operation: str = "train") -> ModelRunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM model_run WHERE system = ? AND version = ? AND operation = ?",
            (system, version, operation),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list(self, system: str | None = None) -> builtins.list[ModelRunRecord]:
        if system is not None:
            rows = self._conn.execute(
                "SELECT * FROM model_run WHERE system = ? ORDER BY created_at DESC",
                (system,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM model_run ORDER BY created_at DESC",
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete(self, system: str, version: str, operation: str = "train") -> None:
        self._conn.execute(
            "DELETE FROM model_run WHERE system = ? AND version = ? AND operation = ?",
            (system, version, operation),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ModelRunRecord:
        return ModelRunRecord(
            id=row["id"],
            system=row["system"],
            version=row["version"],
            operation=row["operation"],
            train_dataset_id=row["train_dataset_id"],
            validation_dataset_id=row["validation_dataset_id"],
            holdout_dataset_id=row["holdout_dataset_id"],
            config_json=json.loads(row["config_json"]),
            metrics_json=json.loads(row["metrics_json"]) if row["metrics_json"] is not None else None,
            artifact_type=row["artifact_type"],
            artifact_path=row["artifact_path"],
            git_commit=row["git_commit"],
            tags_json=json.loads(row["tags_json"]) if row["tags_json"] is not None else None,
            created_at=row["created_at"],
        )
