from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import FeatureCheckpoint, TargetResult
from fantasy_baseball_manager.repos.errors import DuplicateCheckpointError

if TYPE_CHECKING:
    import builtins

    from fantasy_baseball_manager.repos.protocols import ConnectionProvider


class SqliteCheckpointRepo:
    def __init__(self, provider: ConnectionProvider) -> None:
        self._provider = provider

    def save(self, checkpoint: FeatureCheckpoint, *, force: bool = False) -> None:
        with self._provider.connection() as conn:
            target_results_json = {
                target: {
                    "rmse": tr.rmse,
                    "baseline_rmse": tr.baseline_rmse,
                    "delta": tr.delta,
                    "delta_pct": tr.delta_pct,
                }
                for target, tr in checkpoint.target_results.items()
            }
            values = (
                checkpoint.name,
                checkpoint.model,
                checkpoint.player_type,
                json.dumps(checkpoint.feature_columns),
                json.dumps(checkpoint.params),
                json.dumps(target_results_json),
                checkpoint.experiment_id,
                checkpoint.created_at,
                checkpoint.notes,
            )

            if force:
                conn.execute(
                    "DELETE FROM feature_checkpoint WHERE name = ? AND model = ?",
                    (checkpoint.name, checkpoint.model),
                )

            try:
                conn.execute(
                    """INSERT INTO feature_checkpoint
                           (name, model, player_type, feature_columns, params,
                            target_results, experiment_id, created_at, notes)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    values,
                )
            except sqlite3.IntegrityError:
                raise DuplicateCheckpointError(checkpoint.name, checkpoint.model) from None

    def get(self, name: str, model: str) -> FeatureCheckpoint | None:
        with self._provider.connection() as conn:
            row = conn.execute(
                "SELECT * FROM feature_checkpoint WHERE name = ? AND model = ?",
                (name, model),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_checkpoint(row)

    def list(self, model: str | None = None) -> builtins.list[FeatureCheckpoint]:
        with self._provider.connection() as conn:
            if model is not None:
                rows = conn.execute(
                    "SELECT * FROM feature_checkpoint WHERE model = ? ORDER BY name",
                    (model,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM feature_checkpoint ORDER BY name",
                ).fetchall()
            return [self._row_to_checkpoint(row) for row in rows]

    def delete(self, name: str, model: str) -> bool:
        with self._provider.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM feature_checkpoint WHERE name = ? AND model = ?",
                (name, model),
            )
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_checkpoint(row: sqlite3.Row) -> FeatureCheckpoint:
        target_results_raw = json.loads(row["target_results"])
        target_results = {
            target: TargetResult(
                rmse=tr["rmse"],
                baseline_rmse=tr["baseline_rmse"],
                delta=tr["delta"],
                delta_pct=tr["delta_pct"],
            )
            for target, tr in target_results_raw.items()
        }
        return FeatureCheckpoint(
            name=row["name"],
            model=row["model"],
            player_type=row["player_type"],
            feature_columns=json.loads(row["feature_columns"]),
            params=json.loads(row["params"]),
            target_results=target_results,
            experiment_id=row["experiment_id"],
            created_at=row["created_at"],
            notes=row["notes"],
        )
