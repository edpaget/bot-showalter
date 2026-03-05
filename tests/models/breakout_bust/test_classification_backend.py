"""Tests for ClassificationTrainingBackend."""

from typing import Any

import numpy as np

from fantasy_baseball_manager.domain import OutcomeLabel
from fantasy_baseball_manager.domain.model_protocol import (
    FittedModels,
    TrainingBackend,
)
from fantasy_baseball_manager.models.breakout_bust.classification_backend import (
    ClassificationFittedModels,
    ClassificationTrainingBackend,
)


class TestClassificationTrainingBackendProtocol:
    def test_isinstance_check(self) -> None:
        assert isinstance(ClassificationTrainingBackend(), TrainingBackend)


class TestClassificationExtractFeatures:
    def test_none_becomes_nan(self) -> None:
        backend = ClassificationTrainingBackend()
        rows: list[dict[str, Any]] = [
            {"a": 1.0, "b": None},
            {"a": 3.0, "b": 4.0},
        ]
        result = backend.extract_features(rows, ["a", "b"])
        assert result[0][0] == 1.0
        assert np.isnan(result[0][1])
        assert result[1] == [3.0, 4.0]


class TestClassificationExtractTargets:
    def test_extracts_binary_indicators_from_label(self) -> None:
        backend = ClassificationTrainingBackend()
        rows: list[dict[str, Any]] = [
            {"label": OutcomeLabel.BREAKOUT},
            {"label": OutcomeLabel.BUST},
            {"label": OutcomeLabel.NEUTRAL},
        ]
        targets = backend.extract_targets(rows, ["p_breakout", "p_bust"])

        assert "p_breakout" in targets
        assert "p_bust" in targets

        # All rows have labels, so all indices should be present
        assert targets["p_breakout"].indices == [0, 1, 2]
        assert targets["p_bust"].indices == [0, 1, 2]

        # BREAKOUT → p_breakout=1.0, p_bust=0.0
        assert targets["p_breakout"].values == [1.0, 0.0, 0.0]
        assert targets["p_bust"].values == [0.0, 1.0, 0.0]

    def test_skips_rows_without_label(self) -> None:
        backend = ClassificationTrainingBackend()
        rows: list[dict[str, Any]] = [
            {"label": OutcomeLabel.BREAKOUT},
            {"some_field": 1.0},  # no label
            {"label": OutcomeLabel.NEUTRAL},
        ]
        targets = backend.extract_targets(rows, ["p_breakout", "p_bust"])
        assert targets["p_breakout"].indices == [0, 2]
        assert targets["p_breakout"].values == [1.0, 0.0]


class TestClassificationFit:
    def _build_training_data(self) -> tuple[list[list[float]], dict[str, Any]]:
        """Build simple training data with 3 classes."""
        rng = np.random.default_rng(42)
        rows: list[dict[str, Any]] = []
        for _ in range(60):
            a = rng.uniform(0, 1)
            b = rng.uniform(0, 1)
            # Simple rule: high a → breakout, high b → bust, else neutral
            if a > 0.7:
                label = OutcomeLabel.BREAKOUT
            elif b > 0.7:
                label = OutcomeLabel.BUST
            else:
                label = OutcomeLabel.NEUTRAL
            rows.append({"a": a, "b": b, "label": label})

        backend = ClassificationTrainingBackend()
        X = backend.extract_features(rows, ["a", "b"])
        targets = backend.extract_targets(rows, ["p_breakout", "p_bust"])
        return X, targets

    def test_returns_fitted_models(self) -> None:
        backend = ClassificationTrainingBackend()
        X, targets = self._build_training_data()
        fitted = backend.fit(X, targets, {})
        assert isinstance(fitted, FittedModels)
        assert isinstance(fitted, ClassificationFittedModels)

    def test_predict_returns_probabilities(self) -> None:
        backend = ClassificationTrainingBackend()
        X, targets = self._build_training_data()
        fitted = backend.fit(X, targets, {})

        preds = fitted.predict("p_breakout", X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)
        # Probabilities should be between 0 and 1
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)

    def test_score_returns_rmse(self) -> None:
        backend = ClassificationTrainingBackend()
        X, targets = self._build_training_data()
        fitted = backend.fit(X, targets, {})

        scores = fitted.score(X, targets)
        assert "p_breakout" in scores
        assert "p_bust" in scores
        assert scores["p_breakout"] >= 0.0
        assert scores["p_bust"] >= 0.0
