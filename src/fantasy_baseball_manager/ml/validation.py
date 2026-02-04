"""Validation framework for ML models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ValidationMetrics:
    """Metrics from validating on a single fold."""

    fold_name: str
    n_samples: int
    rmse: float
    mae: float
    r_squared: float


@dataclass(frozen=True)
class StatValidationResult:
    """Validation results for a single stat across all folds."""

    stat_name: str
    fold_metrics: tuple[ValidationMetrics, ...]

    @property
    def mean_rmse(self) -> float:
        """Average RMSE across all folds."""
        if not self.fold_metrics:
            return 0.0
        return sum(m.rmse for m in self.fold_metrics) / len(self.fold_metrics)

    @property
    def mean_mae(self) -> float:
        """Average MAE across all folds."""
        if not self.fold_metrics:
            return 0.0
        return sum(m.mae for m in self.fold_metrics) / len(self.fold_metrics)

    @property
    def mean_r_squared(self) -> float:
        """Average R-squared across all folds."""
        if not self.fold_metrics:
            return 0.0
        return sum(m.r_squared for m in self.fold_metrics) / len(self.fold_metrics)

    @property
    def total_samples(self) -> int:
        """Total number of validation samples across all folds."""
        return sum(m.n_samples for m in self.fold_metrics)


@dataclass(frozen=True)
class ValidationReport:
    """Complete validation report for a player type."""

    player_type: str
    strategy_name: str
    stat_results: tuple[StatValidationResult, ...]
    training_years: tuple[int, ...]
    holdout_years: tuple[int, ...]

    def to_dict(self) -> dict:
        """Serialize to a dictionary for persistence."""
        return {
            "player_type": self.player_type,
            "strategy_name": self.strategy_name,
            "stat_results": [
                {
                    "stat_name": sr.stat_name,
                    "fold_metrics": [
                        {
                            "fold_name": m.fold_name,
                            "n_samples": m.n_samples,
                            "rmse": m.rmse,
                            "mae": m.mae,
                            "r_squared": m.r_squared,
                        }
                        for m in sr.fold_metrics
                    ],
                }
                for sr in self.stat_results
            ],
            "training_years": list(self.training_years),
            "holdout_years": list(self.holdout_years),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ValidationReport:
        """Deserialize from a dictionary."""
        stat_results = tuple(
            StatValidationResult(
                stat_name=sr["stat_name"],
                fold_metrics=tuple(
                    ValidationMetrics(
                        fold_name=m["fold_name"],
                        n_samples=m["n_samples"],
                        rmse=m["rmse"],
                        mae=m["mae"],
                        r_squared=m["r_squared"],
                    )
                    for m in sr["fold_metrics"]
                ),
            )
            for sr in data["stat_results"]
        )
        return cls(
            player_type=data["player_type"],
            strategy_name=data["strategy_name"],
            stat_results=stat_results,
            training_years=tuple(data["training_years"]),
            holdout_years=tuple(data["holdout_years"]),
        )


@dataclass(frozen=True)
class ValidationSplit:
    """Definition of a single validation split."""

    fold_name: str
    train_years: tuple[int, ...]
    val_years: tuple[int, ...]


@runtime_checkable
class ValidationStrategy(Protocol):
    """Protocol for validation strategies."""

    @property
    def name(self) -> str:
        """Name of the validation strategy."""
        ...

    def generate_splits(self, available_years: tuple[int, ...]) -> list[ValidationSplit]:
        """Generate validation splits from available years.

        Args:
            available_years: Tuple of years available for training/validation.
                            Should be sorted in ascending order.

        Returns:
            List of ValidationSplit objects defining train/val splits.
        """
        ...


@dataclass(frozen=True)
class TimeSeriesHoldout:
    """Time-series holdout validation strategy.

    Trains on earlier years, validates on later years.
    For example, with holdout_years=1 and available years (2019, 2020, 2021, 2022):
    - Train on: 2019, 2020, 2021
    - Validate on: 2022
    """

    holdout_years: int = 1

    @property
    def name(self) -> str:
        return f"time_series_holdout_{self.holdout_years}y"

    def generate_splits(self, available_years: tuple[int, ...]) -> list[ValidationSplit]:
        """Generate a single train/validation split.

        Args:
            available_years: Years available for training/validation (sorted ascending).

        Returns:
            List containing a single ValidationSplit.

        Raises:
            ValueError: If not enough years available for the holdout.
        """
        if len(available_years) <= self.holdout_years:
            raise ValueError(
                f"Need more than {self.holdout_years} years for holdout, "
                f"got {len(available_years)}"
            )

        sorted_years = tuple(sorted(available_years))
        split_idx = len(sorted_years) - self.holdout_years

        train_years = sorted_years[:split_idx]
        val_years = sorted_years[split_idx:]

        return [
            ValidationSplit(
                fold_name="holdout",
                train_years=train_years,
                val_years=val_years,
            )
        ]


@dataclass(frozen=True)
class LeaveOneYearOut:
    """Leave-one-year-out cross-validation strategy.

    Creates N folds for N years, each holding out one year.
    For example, with years (2019, 2020, 2021):
    - Fold 1: Train on 2020, 2021; Validate on 2019
    - Fold 2: Train on 2019, 2021; Validate on 2020
    - Fold 3: Train on 2019, 2020; Validate on 2021
    """

    @property
    def name(self) -> str:
        return "leave_one_year_out"

    def generate_splits(self, available_years: tuple[int, ...]) -> list[ValidationSplit]:
        """Generate N splits, one for each year.

        Args:
            available_years: Years available for training/validation.

        Returns:
            List of ValidationSplit objects, one per year.

        Raises:
            ValueError: If fewer than 2 years available.
        """
        if len(available_years) < 2:
            raise ValueError(
                f"Need at least 2 years for leave-one-year-out, got {len(available_years)}"
            )

        sorted_years = tuple(sorted(available_years))
        splits: list[ValidationSplit] = []

        for i, holdout_year in enumerate(sorted_years):
            train_years = tuple(y for y in sorted_years if y != holdout_year)
            splits.append(
                ValidationSplit(
                    fold_name=f"fold_{i + 1}_holdout_{holdout_year}",
                    train_years=train_years,
                    val_years=(holdout_year,),
                )
            )

        return splits


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for early stopping during training."""

    enabled: bool = True
    patience: int = 10
    eval_fraction: float = 0.1


def compute_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_name: str,
) -> ValidationMetrics:
    """Compute validation metrics from predictions.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        fold_name: Name of the validation fold.

    Returns:
        ValidationMetrics with RMSE, MAE, and R-squared.

    Raises:
        ValueError: If arrays are empty or have mismatched lengths.
    """
    import numpy as np

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Array length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}"
        )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # RMSE
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    # MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R-squared
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    # R² is 0 when all targets are the same value (ss_tot == 0)
    r_squared = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot

    return ValidationMetrics(
        fold_name=fold_name,
        n_samples=len(y_true),
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
    )
