"""GBM-specific implementation of the TrainingBackend protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fantasy_baseball_manager.models.gbm_training import (
    extract_features as gbm_extract_features,
)
from fantasy_baseball_manager.models.gbm_training import (
    extract_targets as gbm_extract_targets,
)
from fantasy_baseball_manager.models.gbm_training import (
    fit_models as gbm_fit_models,
)
from fantasy_baseball_manager.models.sampling import holdout_metrics

if TYPE_CHECKING:
    from sklearn.ensemble import HistGradientBoostingRegressor

    from fantasy_baseball_manager.domain import TargetVector


class GBMFittedModels:
    """Wraps a dict of fitted HistGradientBoostingRegressor models."""

    def __init__(self, models: dict[str, HistGradientBoostingRegressor]) -> None:
        self._models = models

    def predict(self, target: str, X: list[list[float]]) -> np.ndarray:
        return self._models[target].predict(X)

    def score(self, X: list[list[float]], targets: dict[str, TargetVector]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for target_name, model in self._models.items():
            tv = targets[target_name]
            X_filtered = [X[i] for i in tv.indices]
            y_pred = model.predict(X_filtered)
            target_metrics = holdout_metrics(np.array(tv.values), y_pred)
            metrics[target_name] = target_metrics["rmse"]
        return metrics


class GBMTrainingBackend:
    """TrainingBackend implementation backed by HistGradientBoostingRegressor."""

    def extract_features(self, rows: list[dict[str, Any]], columns: list[str]) -> list[list[float]]:
        return gbm_extract_features(rows, columns)

    def extract_targets(self, rows: list[dict[str, Any]], targets: list[str]) -> dict[str, TargetVector]:
        return gbm_extract_targets(rows, targets)

    def fit(self, X: list[list[float]], targets: dict[str, TargetVector], params: dict[str, Any]) -> GBMFittedModels:
        models = gbm_fit_models(X, targets, params)
        return GBMFittedModels(models)
