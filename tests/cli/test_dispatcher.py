from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from fantasy_baseball_manager.cli._dispatcher import dispatch, UnsupportedOperation
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain.model_run import ModelRunRecord
from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.models.protocols import (
    EvalResult,
    ModelConfig,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import _clear, register
from fantasy_baseball_manager.models.run_manager import RunManager


def _make_assembler() -> SqliteDatasetAssembler:
    conn = create_connection(":memory:")
    return SqliteDatasetAssembler(conn)


class _FakePreparableOnly:
    @property
    def name(self) -> str:
        return "fake"

    @property
    def description(self) -> str:
        return "Fake model"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare"})

    @property
    def artifact_type(self) -> str:
        return "none"

    def prepare(self, config: ModelConfig, assembler: DatasetAssembler) -> PrepareResult:
        return PrepareResult(model_name="fake", rows_processed=42, artifacts_path="/tmp")


class _FakeFullModel(_FakePreparableOnly):
    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "evaluate"})

    def train(self, config: ModelConfig) -> TrainResult:
        return TrainResult(model_name="fake", metrics={"rmse": 0.5}, artifacts_path="/tmp")

    def evaluate(self, config: ModelConfig) -> EvalResult:
        return EvalResult(model_name="fake", metrics={"mae": 0.3})


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _clear()


class TestDispatch:
    def test_dispatch_prepare(self) -> None:
        register("fake")(_FakePreparableOnly)
        assembler = _make_assembler()
        result = dispatch("prepare", "fake", ModelConfig(), assembler=assembler)
        assert isinstance(result, PrepareResult)
        assert result.rows_processed == 42

    def test_dispatch_train(self) -> None:
        register("fake")(_FakeFullModel)
        result = dispatch("train", "fake", ModelConfig())
        assert isinstance(result, TrainResult)
        assert result.metrics == {"rmse": 0.5}

    def test_dispatch_evaluate(self) -> None:
        register("fake")(_FakeFullModel)
        result = dispatch("evaluate", "fake", ModelConfig())
        assert isinstance(result, EvalResult)

    def test_unsupported_operation_raises(self) -> None:
        register("fake")(_FakePreparableOnly)
        with pytest.raises(UnsupportedOperation, match="does not support 'train'"):
            dispatch("train", "fake", ModelConfig())

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError, match="no model registered"):
            dispatch("train", "nonexistent", ModelConfig())

    def test_unknown_operation_raises(self) -> None:
        register("fake")(_FakePreparableOnly)
        with pytest.raises(UnsupportedOperation, match="does not support 'bogus'"):
            dispatch("bogus", "fake", ModelConfig())


class _FakeModelRunRepo:
    def __init__(self) -> None:
        self._records: list[ModelRunRecord] = []
        self._next_id = 1

    def upsert(self, record: ModelRunRecord) -> int:
        row_id = self._next_id
        self._next_id += 1
        self._records.append(record)
        return row_id

    def get(self, system: str, version: str) -> ModelRunRecord | None:
        return None

    def list(self, system: str | None = None) -> builtins.list[ModelRunRecord]:
        return builtins.list(self._records)

    def delete(self, system: str, version: str) -> None:
        pass


class TestDispatchWithRunManager:
    def test_dispatch_train_with_run_manager(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )

        register("fake")(_FakeFullModel)
        repo = _FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")

        result = dispatch("train", "fake", config, run_manager=mgr)

        assert isinstance(result, TrainResult)
        assert len(repo._records) == 1
        assert repo._records[0].system == "fake"
        assert repo._records[0].version == "v1"

    def test_dispatch_non_train_ignores_run_manager(self, tmp_path: Path) -> None:
        register("fake")(_FakeFullModel)
        repo = _FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")

        result = dispatch("evaluate", "fake", config, run_manager=mgr)

        assert isinstance(result, EvalResult)
        assert len(repo._records) == 0
