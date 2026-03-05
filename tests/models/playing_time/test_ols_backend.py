"""Tests for OLSTrainingBackend."""

from typing import Any

import numpy as np

from fantasy_baseball_manager.domain.model_protocol import (
    FittedModels,
    TrainingBackend,
)
from fantasy_baseball_manager.models.playing_time.ols_backend import (
    OLSFittedModels,
    OLSTrainingBackend,
)


class TestOLSTrainingBackendProtocol:
    def test_isinstance_check(self) -> None:
        assert isinstance(OLSTrainingBackend(), TrainingBackend)


class TestOLSExtractFeatures:
    def test_none_becomes_zero(self) -> None:
        backend = OLSTrainingBackend()
        rows: list[dict[str, Any]] = [
            {"a": 1.0, "b": None},
            {"a": 3.0, "b": 4.0},
        ]
        result = backend.extract_features(rows, ["a", "b"])
        assert result[0] == [1.0, 0.0]
        assert result[1] == [3.0, 4.0]


class TestOLSExtractTargets:
    def test_extracts_target_column(self) -> None:
        backend = OLSTrainingBackend()
        rows: list[dict[str, Any]] = [
            {"target_pa": 500.0},
            {"target_pa": 600.0},
            {"target_pa": None},
        ]
        targets = backend.extract_targets(rows, ["pa"])
        assert "pa" in targets
        assert targets["pa"].indices == [0, 1]
        assert targets["pa"].values == [500.0, 600.0]

    def test_missing_target_column_skipped(self) -> None:
        backend = OLSTrainingBackend()
        rows: list[dict[str, Any]] = [{"other": 1.0}]
        targets = backend.extract_targets(rows, ["pa"])
        assert targets["pa"].indices == []
        assert targets["pa"].values == []


class TestOLSFit:
    def _build_training_data(self) -> tuple[list[list[float]], dict[str, Any]]:
        """Build simple linear training data: target_pa = 100*a + 50*b."""
        backend = OLSTrainingBackend()
        rows: list[dict[str, Any]] = []
        rng = np.random.default_rng(42)
        for _ in range(50):
            a = rng.uniform(1, 5)
            b = rng.uniform(0, 3)
            rows.append({"a": a, "b": b, "target_pa": 100 * a + 50 * b + rng.normal(0, 5)})
        X = backend.extract_features(rows, ["a", "b"])
        targets = backend.extract_targets(rows, ["pa"])
        return X, targets

    def test_returns_fitted_models(self) -> None:
        backend = OLSTrainingBackend()
        X, targets = self._build_training_data()
        fitted = backend.fit(X, targets, {})
        assert isinstance(fitted, FittedModels)
        assert isinstance(fitted, OLSFittedModels)

    def test_predict_returns_predictions(self) -> None:
        backend = OLSTrainingBackend()
        X, targets = self._build_training_data()
        fitted = backend.fit(X, targets, {})
        preds = fitted.predict("pa", X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)

    def test_predictions_are_reasonable(self) -> None:
        backend = OLSTrainingBackend()
        X, targets = self._build_training_data()
        fitted = backend.fit(X, targets, {})
        scores = fitted.score(X, targets)
        assert "pa" in scores
        # RMSE should be small for a near-linear relationship
        assert scores["pa"] < 20.0
