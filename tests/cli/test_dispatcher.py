from __future__ import annotations

import builtins
import subprocess
from pathlib import Path

import pytest

from fantasy_baseball_manager.cli._dispatcher import dispatch, UnsupportedOperation
from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.domain.model_run import ModelRunRecord
from fantasy_baseball_manager.models.protocols import (
    ModelConfig,
    PredictResult,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.run_manager import RunManager


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

    def prepare(self, config: ModelConfig) -> PrepareResult:
        return PrepareResult(model_name="fake", rows_processed=42, artifacts_path="/tmp")


class _FakeFullModel(_FakePreparableOnly):
    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "evaluate", "predict"})

    def train(self, config: ModelConfig) -> TrainResult:
        return TrainResult(model_name="fake", metrics={"rmse": 0.5}, artifacts_path="/tmp")

    def evaluate(self, config: ModelConfig) -> SystemMetrics:
        return SystemMetrics(system="fake", version="latest", source_type="first_party", metrics={})

    def predict(self, config: ModelConfig) -> PredictResult:
        return PredictResult(model_name="fake", predictions=[{"hr": 30}], output_path="/tmp")


class TestDispatch:
    def test_dispatch_prepare(self) -> None:
        result = dispatch("prepare", _FakePreparableOnly(), ModelConfig())
        assert isinstance(result, PrepareResult)
        assert result.rows_processed == 42

    def test_dispatch_train(self) -> None:
        result = dispatch("train", _FakeFullModel(), ModelConfig())
        assert isinstance(result, TrainResult)
        assert result.metrics == {"rmse": 0.5}

    def test_dispatch_evaluate(self) -> None:
        result = dispatch("evaluate", _FakeFullModel(), ModelConfig())
        assert isinstance(result, SystemMetrics)

    def test_unsupported_operation_raises(self) -> None:
        with pytest.raises(UnsupportedOperation, match="does not support 'train'"):
            dispatch("train", _FakePreparableOnly(), ModelConfig())

    def test_unknown_operation_raises(self) -> None:
        with pytest.raises(UnsupportedOperation, match="does not support 'bogus'"):
            dispatch("bogus", _FakePreparableOnly(), ModelConfig())


class _FakeModelRunRepo:
    def __init__(self) -> None:
        self._records: list[ModelRunRecord] = []
        self._next_id = 1

    def upsert(self, record: ModelRunRecord) -> int:
        row_id = self._next_id
        self._next_id += 1
        self._records.append(record)
        return row_id

    def get(self, system: str, version: str, operation: str = "train") -> ModelRunRecord | None:
        return None

    def list(self, system: str | None = None) -> builtins.list[ModelRunRecord]:
        return builtins.list(self._records)

    def delete(self, system: str, version: str, operation: str = "train") -> None:
        pass


class TestDispatchWithRunManager:
    def test_dispatch_train_with_run_manager(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )

        repo = _FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")

        result = dispatch("train", _FakeFullModel(), config, run_manager=mgr)

        assert isinstance(result, TrainResult)
        assert len(repo._records) == 1
        assert repo._records[0].system == "fake"
        assert repo._records[0].version == "v1"

    def test_dispatch_train_transfers_metrics_to_run_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )

        repo = _FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")

        dispatch("train", _FakeFullModel(), config, run_manager=mgr)

        assert repo._records[0].metrics_json == {"rmse": 0.5}

    def test_dispatch_predict_with_run_manager(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=1),
        )

        repo = _FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")

        result = dispatch("predict", _FakeFullModel(), config, run_manager=mgr)

        assert isinstance(result, PredictResult)
        assert len(repo._records) == 1
        assert repo._records[0].system == "fake"
        assert repo._records[0].version == "v1"
        assert repo._records[0].operation == "predict"

    def test_dispatch_non_tracked_operation_ignores_run_manager(self, tmp_path: Path) -> None:
        repo = _FakeModelRunRepo()
        mgr = RunManager(model_run_repo=repo, artifacts_root=tmp_path)
        config = ModelConfig(version="v1")

        result = dispatch("evaluate", _FakeFullModel(), config, run_manager=mgr)

        assert isinstance(result, SystemMetrics)
        assert len(repo._records) == 0
