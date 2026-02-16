"""Shared splitting utilities for model training and evaluation."""

from collections.abc import Iterator
from typing import Any


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
