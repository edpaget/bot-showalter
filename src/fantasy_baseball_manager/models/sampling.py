"""Shared splitting and evaluation utilities for model training."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray


def holdout_metrics(
    y_actual: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> dict[str, float]:
    """Compute R², RMSE, and sample count for holdout evaluation.

    Returns a dict with keys ``r_squared``, ``rmse``, and ``n``.
    Empty arrays return all-zero values.
    """
    n = len(y_actual)
    if n == 0:
        return {"r_squared": 0.0, "rmse": 0.0, "n": 0}

    ss_res = float(np.sum((y_actual - y_pred) ** 2))
    ss_tot = float(np.sum((y_actual - np.mean(y_actual)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    rmse = float(np.sqrt(ss_res / n))
    return {"r_squared": r_squared, "rmse": rmse, "n": n}


def temporal_holdout_split(
    rows: list[dict[str, Any]],
    season_column: str = "season",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train/holdout by holding out the most recent season.

    Returns (train_rows, holdout_rows) where holdout contains only the
    maximum season value and train contains all earlier seasons.

    Raises ValueError if fewer than 2 distinct seasons are present.
    """
    seasons = {row[season_column] for row in rows}
    if len(seasons) < 2:
        msg = f"temporal_holdout_split requires at least 2 distinct seasons (got {len(seasons)})"
        raise ValueError(msg)

    max_season = max(seasons)
    train = [row for row in rows if row[season_column] != max_season]
    holdout = [row for row in rows if row[season_column] == max_season]
    return train, holdout


def temporal_expanding_cv(
    seasons: list[int],
) -> Iterator[tuple[list[int], int]]:
    """Yield (train_seasons, test_season) with expanding training window.

    For seasons [2022, 2023, 2024]:
      Fold 0: train=[2022], test=2023
      Fold 1: train=[2022, 2023], test=2024

    Requires at least 3 seasons (to get at least 1 fold, leaving final season
    for holdout). Raises ValueError otherwise.
    """
    if len(seasons) < 3:
        msg = f"temporal_expanding_cv requires at least 3 seasons (got {len(seasons)})"
        raise ValueError(msg)

    sorted_seasons = sorted(seasons)
    # Last season is reserved for holdout; iterate up to second-to-last
    for i in range(1, len(sorted_seasons) - 1):
        train_seasons = sorted_seasons[:i]
        test_season = sorted_seasons[i]
        yield train_seasons, test_season


def season_kfold(
    rows: list[dict[str, Any]],
    n_folds: int = 5,
    season_column: str = "season",
) -> Iterator[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Yield (train, test) splits using season-based k-fold cross-validation.

    Each fold holds out one or more seasons for testing and trains on the rest.
    Seasons are assigned to folds via round-robin on sorted order. If fewer
    distinct seasons exist than *n_folds*, the number of folds is capped at
    the season count. A single season yields zero folds.
    """
    seasons = sorted({row[season_column] for row in rows})
    if len(seasons) <= 1:
        return

    actual_folds = min(n_folds, len(seasons))
    fold_for_season: dict[Any, int] = {s: i % actual_folds for i, s in enumerate(seasons)}

    for fold_idx in range(actual_folds):
        test = [row for row in rows if fold_for_season[row[season_column]] == fold_idx]
        train = [row for row in rows if fold_for_season[row[season_column]] != fold_idx]
        yield train, test
