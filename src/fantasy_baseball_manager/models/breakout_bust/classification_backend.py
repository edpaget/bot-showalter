"""Classification-specific implementation of the TrainingBackend protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from fantasy_baseball_manager.domain import OutcomeLabel, TargetVector
from fantasy_baseball_manager.models.gbm_training import (
    extract_features as gbm_extract_features,
)
from fantasy_baseball_manager.models.sampling import holdout_metrics

# Duplicated from model.py to avoid circular import.
_LABEL_TO_INT: dict[OutcomeLabel, int] = {
    OutcomeLabel.NEUTRAL: 0,
    OutcomeLabel.BREAKOUT: 1,
    OutcomeLabel.BUST: 2,
}

# Map target names to the class integer whose probability they represent.
_TARGET_TO_CLASS: dict[str, int] = {
    "p_breakout": _LABEL_TO_INT[OutcomeLabel.BREAKOUT],
    "p_bust": _LABEL_TO_INT[OutcomeLabel.BUST],
}


class ClassificationFittedModels:
    """Wraps a fitted HistGradientBoostingClassifier for probability predictions."""

    def __init__(
        self,
        classifier: HistGradientBoostingClassifier,
        class_to_col: dict[int, int],
    ) -> None:
        self._clf = classifier
        self._class_to_col = class_to_col  # maps class int → column index in predict_proba

    def predict(self, target: str, X: list[list[float]]) -> np.ndarray:
        proba = self._clf.predict_proba(X)
        class_int = _TARGET_TO_CLASS[target]
        col_idx = self._class_to_col[class_int]
        return proba[:, col_idx]

    def score(self, X: list[list[float]], targets: dict[str, TargetVector]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for target_name in targets:
            tv = targets[target_name]
            X_filtered = [X[i] for i in tv.indices]
            y_pred = self.predict(target_name, X_filtered)
            m = holdout_metrics(np.array(tv.values), y_pred)
            metrics[target_name] = m["rmse"]
        return metrics


class ClassificationTrainingBackend:
    """TrainingBackend implementation for classification models (breakout-bust)."""

    def extract_features(self, rows: list[dict[str, Any]], columns: list[str]) -> list[list[float]]:
        return gbm_extract_features(rows, columns)

    def extract_targets(self, rows: list[dict[str, Any]], targets: list[str]) -> dict[str, TargetVector]:
        result: dict[str, TargetVector] = {t: TargetVector([], []) for t in targets}
        for i, row in enumerate(rows):
            label = row.get("label")
            if label is None:
                continue
            label_int = _LABEL_TO_INT[label]
            for target in targets:
                class_int = _TARGET_TO_CLASS[target]
                indicator = 1.0 if label_int == class_int else 0.0
                result[target].indices.append(i)
                result[target].values.append(indicator)
        return result

    def fit(
        self,
        X: list[list[float]],
        targets: dict[str, TargetVector],
        params: dict[str, Any],
    ) -> ClassificationFittedModels:
        # Reconstruct multiclass labels from the binary indicators.
        # Use p_breakout target vector to get valid row indices (all targets share same indices).
        first_target = next(iter(targets))
        tv = targets[first_target]
        indices = tv.indices

        X_filtered = [X[i] for i in indices]

        # Reconstruct integer labels from indicators across targets.
        n = len(indices)
        y = [_LABEL_TO_INT[OutcomeLabel.NEUTRAL]] * n  # default to neutral
        for target_name, target_tv in targets.items():
            class_int = _TARGET_TO_CLASS[target_name]
            for j, val in enumerate(target_tv.values):
                if val == 1.0:
                    y[j] = class_int

        clf = HistGradientBoostingClassifier(
            max_iter=params.get("max_iter", 200),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
            min_samples_leaf=params.get("min_samples_leaf", 10),
        )
        clf.fit(X_filtered, y)

        # Build class → column index mapping
        class_to_col = {int(c): i for i, c in enumerate(clf.classes_)}

        return ClassificationFittedModels(clf, class_to_col)
