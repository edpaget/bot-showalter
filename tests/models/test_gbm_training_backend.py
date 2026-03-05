"""Tests for GBMTrainingBackend."""

from typing import Any

import numpy as np

from fantasy_baseball_manager.domain.model_protocol import (
    FittedModels,
    TrainingBackend,
)
from fantasy_baseball_manager.models.gbm_training_backend import GBMTrainingBackend


class TestGBMTrainingBackendProtocol:
    def test_isinstance_check(self) -> None:
        assert isinstance(GBMTrainingBackend(), TrainingBackend)


class TestGBMTrainingBackendExtractFeatures:
    def test_delegates_correctly(self) -> None:
        backend = GBMTrainingBackend()
        rows: list[dict[str, Any]] = [
            {"a": 1.0, "b": 2.0},
            {"a": 3.0, "b": 4.0},
        ]
        result = backend.extract_features(rows, ["a", "b"])
        assert result == [[1.0, 2.0], [3.0, 4.0]]


class TestGBMTrainingBackendExtractTargets:
    def test_delegates_correctly(self) -> None:
        backend = GBMTrainingBackend()
        rows: list[dict[str, Any]] = [
            {"target_slg": 0.400},
            {"target_slg": 0.500},
        ]
        result = backend.extract_targets(rows, ["slg"])
        assert "slg" in result
        assert result["slg"].values == [0.400, 0.500]


class TestGBMTrainingBackendFit:
    def test_returns_fitted_models(self) -> None:
        backend = GBMTrainingBackend()
        # Build simple training data
        rows: list[dict[str, Any]] = []
        for i in range(50):
            a = (i % 10) * 0.1
            b = ((i + 3) % 10) * 0.1
            rows.append({"a": a, "b": b, "target_slg": a * 0.5 + b * 0.3})

        X = backend.extract_features(rows, ["a", "b"])
        targets = backend.extract_targets(rows, ["slg"])
        fitted = backend.fit(X, targets, {})

        assert isinstance(fitted, FittedModels)

    def test_fitted_predict(self) -> None:
        backend = GBMTrainingBackend()
        rows: list[dict[str, Any]] = []
        for i in range(50):
            a = (i % 10) * 0.1
            b = ((i + 3) % 10) * 0.1
            rows.append({"a": a, "b": b, "target_slg": a * 0.5 + b * 0.3})

        X = backend.extract_features(rows, ["a", "b"])
        targets = backend.extract_targets(rows, ["slg"])
        fitted = backend.fit(X, targets, {})

        predictions = fitted.predict("slg", X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 50

    def test_fitted_score(self) -> None:
        backend = GBMTrainingBackend()
        rows: list[dict[str, Any]] = []
        for i in range(50):
            a = (i % 10) * 0.1
            b = ((i + 3) % 10) * 0.1
            rows.append({"a": a, "b": b, "target_slg": a * 0.5 + b * 0.3})

        X = backend.extract_features(rows, ["a", "b"])
        targets = backend.extract_targets(rows, ["slg"])
        fitted = backend.fit(X, targets, {})

        scores = fitted.score(X, targets)
        assert "slg" in scores
        assert scores["slg"] >= 0.0
