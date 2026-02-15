import inspect
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.models.protocols import ModelConfig, ProjectionModel
from fantasy_baseball_manager.models.registry import get
from fantasy_baseball_manager.models.run_manager import RunManager
from fantasy_baseball_manager.repos.batting_stats_repo import SqliteBattingStatsRepo
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo
from fantasy_baseball_manager.repos.pitching_stats_repo import SqlitePitchingStatsRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from fantasy_baseball_manager.services.projection_evaluator import ProjectionEvaluator


def create_model(name: str, **kwargs: Any) -> ProjectionModel:
    """Look up a model class by name and instantiate it, forwarding matching kwargs."""
    cls = get(name)
    sig = inspect.signature(cls)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**filtered)


@dataclass(frozen=True)
class ModelContext:
    conn: sqlite3.Connection
    model: ProjectionModel
    run_manager: RunManager | None


@contextmanager
def build_model_context(model_name: str, config: ModelConfig) -> Iterator[ModelContext]:
    """Composition-root context manager: opens DB, wires assembler + model, yields context, closes DB."""
    conn = create_connection(Path(config.data_dir) / "fbm.db")
    try:
        assembler = SqliteDatasetAssembler(conn)
        model = create_model(model_name, assembler=assembler)

        run_manager: RunManager | None = None
        if config.version is not None:
            repo = SqliteModelRunRepo(conn)
            run_manager = RunManager(model_run_repo=repo, artifacts_root=Path(config.artifacts_dir))

        yield ModelContext(conn=conn, model=model, run_manager=run_manager)
    finally:
        conn.close()


@dataclass(frozen=True)
class EvalContext:
    conn: sqlite3.Connection
    evaluator: ProjectionEvaluator


@contextmanager
def build_eval_context(data_dir: str) -> Iterator[EvalContext]:
    """Composition-root context manager for eval/compare commands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        evaluator = ProjectionEvaluator(
            SqliteProjectionRepo(conn),
            SqliteBattingStatsRepo(conn),
            SqlitePitchingStatsRepo(conn),
        )
        yield EvalContext(conn=conn, evaluator=evaluator)
    finally:
        conn.close()


@dataclass(frozen=True)
class RunsContext:
    conn: sqlite3.Connection
    repo: SqliteModelRunRepo


@contextmanager
def build_runs_context(data_dir: str) -> Iterator[RunsContext]:
    """Composition-root context manager for runs subcommands."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield RunsContext(conn=conn, repo=SqliteModelRunRepo(conn))
    finally:
        conn.close()


@dataclass(frozen=True)
class ImportContext:
    conn: sqlite3.Connection
    player_repo: SqlitePlayerRepo
    proj_repo: SqliteProjectionRepo
    log_repo: SqliteLoadLogRepo


@contextmanager
def build_import_context(data_dir: str) -> Iterator[ImportContext]:
    """Composition-root context manager for the import command."""
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        yield ImportContext(
            conn=conn,
            player_repo=SqlitePlayerRepo(conn),
            proj_repo=SqliteProjectionRepo(conn),
            log_repo=SqliteLoadLogRepo(conn),
        )
    finally:
        conn.close()
