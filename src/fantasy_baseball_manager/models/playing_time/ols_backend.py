"""OLS/Ridge-specific implementation of the TrainingBackend protocol."""

from typing import Any

import numpy as np

from fantasy_baseball_manager.domain import TargetVector
from fantasy_baseball_manager.models.playing_time.engine import PlayingTimeCoefficients, fit_playing_time
from fantasy_baseball_manager.models.sampling import holdout_metrics


class OLSFittedModels:
    """Wraps fitted OLS/Ridge coefficients for prediction."""

    def __init__(self, models: dict[str, PlayingTimeCoefficients]) -> None:
        self._models = models

    def predict(self, target: str, X: list[list[float]]) -> np.ndarray:
        coeff = self._models[target]
        arr = np.array(X, dtype=np.float64)
        betas = np.array(coeff.coefficients, dtype=np.float64)
        return arr @ betas + coeff.intercept

    def score(self, X: list[list[float]], targets: dict[str, TargetVector]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for target_name in targets:
            tv = targets[target_name]
            X_filtered = [X[i] for i in tv.indices]
            y_pred = self.predict(target_name, X_filtered)
            m = holdout_metrics(np.array(tv.values), y_pred)
            metrics[target_name] = m["rmse"]
        return metrics


class OLSTrainingBackend:
    """TrainingBackend implementation backed by OLS/Ridge regression."""

    def extract_features(self, rows: list[dict[str, Any]], columns: list[str]) -> list[list[float]]:
        matrix: list[list[float]] = []
        for row in rows:
            vector: list[float] = []
            for col in columns:
                value = row.get(col)
                vector.append(float(value) if value is not None else 0.0)
            matrix.append(vector)
        return matrix

    def extract_targets(self, rows: list[dict[str, Any]], targets: list[str]) -> dict[str, TargetVector]:
        result: dict[str, TargetVector] = {t: TargetVector([], []) for t in targets}
        for i, row in enumerate(rows):
            for target in targets:
                value = row.get(f"target_{target}")
                if value is None:
                    continue
                result[target].indices.append(i)
                result[target].values.append(float(value))
        return result

    def fit(
        self,
        X: list[list[float]],
        targets: dict[str, TargetVector],
        params: dict[str, Any],
    ) -> OLSFittedModels:
        alpha = params.get("alpha", 0.0)
        models: dict[str, PlayingTimeCoefficients] = {}

        for target_name, tv in targets.items():
            # Reconstruct rows as dicts for fit_playing_time
            # Use synthetic column names col_0, col_1, ...
            n_cols = len(X[0]) if X else 0
            col_names = [f"__col_{j}" for j in range(n_cols)]
            target_col = f"target_{target_name}"

            rows: list[dict[str, Any]] = []
            for pos, idx in enumerate(tv.indices):
                row: dict[str, Any] = {}
                for j, col in enumerate(col_names):
                    row[col] = X[idx][j]
                row[target_col] = tv.values[pos]
                rows.append(row)

            coeff = fit_playing_time(rows, col_names, target_col, "batter", alpha=alpha)
            models[target_name] = coeff

        return OLSFittedModels(models)
