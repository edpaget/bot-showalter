"""Tests for the TrainingBackend and FittedModels protocols."""

from typing import Any

import numpy as np

from fantasy_baseball_manager.domain.model_protocol import (
    FittedModels,
    TargetVector,
    TrainingBackend,
)


class _ValidFittedModels:
    def predict(self, target: str, X: list[list[float]]) -> np.ndarray:
        return np.zeros(len(X))

    def score(self, X: list[list[float]], targets: dict[str, TargetVector]) -> dict[str, float]:
        return {t: 0.0 for t in targets}


class _ValidTrainingBackend:
    def extract_features(self, rows: list[dict[str, Any]], columns: list[str]) -> list[list[float]]:
        return [[0.0] * len(columns)] * len(rows)

    def extract_targets(self, rows: list[dict[str, Any]], targets: list[str]) -> dict[str, TargetVector]:
        return {t: TargetVector(indices=[], values=[]) for t in targets}

    def fit(self, X: list[list[float]], targets: dict[str, TargetVector], params: dict[str, Any]) -> _ValidFittedModels:
        return _ValidFittedModels()


class _MissingFit:
    def extract_features(self, rows: list[dict[str, Any]], columns: list[str]) -> list[list[float]]:
        return []

    def extract_targets(self, rows: list[dict[str, Any]], targets: list[str]) -> dict[str, TargetVector]:
        return {}


class TestTrainingBackendProtocol:
    def test_satisfying_class_passes_isinstance(self) -> None:
        assert isinstance(_ValidTrainingBackend(), TrainingBackend)

    def test_non_satisfying_class_fails_isinstance(self) -> None:
        assert not isinstance(_MissingFit(), TrainingBackend)

    def test_plain_object_fails_isinstance(self) -> None:
        assert not isinstance(object(), TrainingBackend)


class TestFittedModelsProtocol:
    def test_satisfying_class_passes_isinstance(self) -> None:
        assert isinstance(_ValidFittedModels(), FittedModels)

    def test_plain_object_fails_isinstance(self) -> None:
        assert not isinstance(object(), FittedModels)
