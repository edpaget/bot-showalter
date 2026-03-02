from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import Experiment, TargetResult

if TYPE_CHECKING:
    import builtins
    import sqlite3


@dataclass(frozen=True)
class ExperimentFilter:
    model: str | None = None
    player_type: str | None = None
    tag: str | None = None
    since: str | None = None
    until: str | None = None
    parent_id: int | None = field(default=None, repr=False)


class SqliteExperimentRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, experiment: Experiment) -> int:
        target_results_json = {
            target: {
                "rmse": tr.rmse,
                "baseline_rmse": tr.baseline_rmse,
                "delta": tr.delta,
                "delta_pct": tr.delta_pct,
            }
            for target, tr in experiment.target_results.items()
        }
        cursor = self._conn.execute(
            """INSERT INTO experiment
                   (timestamp, hypothesis, model, player_type, feature_diff,
                    seasons, params, target_results, conclusion, tags, parent_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment.timestamp,
                experiment.hypothesis,
                experiment.model,
                experiment.player_type,
                json.dumps(experiment.feature_diff),
                json.dumps(experiment.seasons),
                json.dumps(experiment.params),
                json.dumps(target_results_json),
                experiment.conclusion,
                json.dumps(experiment.tags),
                experiment.parent_id,
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def get(self, experiment_id: int) -> Experiment | None:
        row = self._conn.execute(
            "SELECT * FROM experiment WHERE id = ?",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_experiment(row)

    def list(self, filter: ExperimentFilter | None = None) -> builtins.list[Experiment]:
        clauses: builtins.list[str] = []
        params: builtins.list[object] = []

        if filter is not None:
            if filter.model is not None:
                clauses.append("model = ?")
                params.append(filter.model)
            if filter.player_type is not None:
                clauses.append("player_type = ?")
                params.append(filter.player_type)
            if filter.tag is not None:
                clauses.append("EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = ?)")
                params.append(filter.tag)
            if filter.since is not None:
                clauses.append("timestamp >= ?")
                params.append(filter.since)
            if filter.until is not None:
                clauses.append("timestamp <= ?")
                params.append(filter.until)
            if filter.parent_id is not None:
                clauses.append("parent_id = ?")
                params.append(filter.parent_id)

        where = " AND ".join(clauses)
        query = "SELECT * FROM experiment"
        if where:
            query += f" WHERE {where}"
        query += " ORDER BY timestamp DESC"

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_experiment(row) for row in rows]

    def delete(self, experiment_id: int) -> None:
        self._conn.execute("DELETE FROM experiment WHERE id = ?", (experiment_id,))

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
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
        return Experiment(
            id=row["id"],
            timestamp=row["timestamp"],
            hypothesis=row["hypothesis"],
            model=row["model"],
            player_type=row["player_type"],
            feature_diff=json.loads(row["feature_diff"]),
            seasons=json.loads(row["seasons"]),
            params=json.loads(row["params"]),
            target_results=target_results,
            conclusion=row["conclusion"],
            tags=json.loads(row["tags"]),
            parent_id=row["parent_id"],
        )
