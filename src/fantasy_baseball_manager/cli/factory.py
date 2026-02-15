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
from fantasy_baseball_manager.repos.model_run_repo import SqliteModelRunRepo


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
