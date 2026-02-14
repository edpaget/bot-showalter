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
                   (system, version, train_dataset_id, validation_dataset_id,
                    holdout_dataset_id, config_json, metrics_json, artifact_type,
                    artifact_path, git_commit, tags_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(system, version) DO UPDATE SET
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
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get(self, system: str, version: str) -> ModelRunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM model_run WHERE system = ? AND version = ?",
            (system, version),
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

    def delete(self, system: str, version: str) -> None:
        self._conn.execute(
            "DELETE FROM model_run WHERE system = ? AND version = ?",
            (system, version),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_record(row: tuple) -> ModelRunRecord:
        return ModelRunRecord(
            id=row[0],
            system=row[1],
            version=row[2],
            train_dataset_id=row[3],
            validation_dataset_id=row[4],
            holdout_dataset_id=row[5],
            config_json=json.loads(row[6]),
            metrics_json=json.loads(row[7]) if row[7] is not None else None,
            artifact_type=row[8],
            artifact_path=row[9],
            git_commit=row[10],
            tags_json=json.loads(row[11]) if row[11] is not None else None,
            created_at=row[12],
        )
