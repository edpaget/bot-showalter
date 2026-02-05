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
            raise ValueError(f"Need more than {self.holdout_years} years for holdout, " f"got {len(available_years)}")

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
            raise ValueError(f"Need at least 2 years for leave-one-year-out, got {len(available_years)}")

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
        raise ValueError(f"Array length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")

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


# =============================================================================
# Holdout Evaluation Framework
# =============================================================================


@dataclass(frozen=True)
class HoldoutEvaluation:
    """Evaluation metrics for a single stat on a held-out test set."""

    stat_name: str
    n_samples: int
    rmse: float
    weighted_rmse: float
    mae: float
    spearman_rho: float


@dataclass(frozen=True)
class BaselineComparison:
    """Comparison of model predictions vs a baseline."""

    stat_name: str
    model_rmse: float
    baseline_rmse: float
    rmse_improvement: float
    rmse_improvement_pct: float
    model_spearman: float
    baseline_spearman: float


@dataclass(frozen=True)
class HoldoutEvaluationReport:
    """Complete holdout evaluation report for a model."""

    model_name: str
    test_years: tuple[int, ...]
    n_samples: int
    stat_evaluations: tuple[HoldoutEvaluation, ...]
    baseline_comparisons: tuple[BaselineComparison, ...] | None = None

    def to_dict(self) -> dict:
        """Serialize to a dictionary for persistence."""
        result: dict = {
            "model_name": self.model_name,
            "test_years": list(self.test_years),
            "n_samples": self.n_samples,
            "stat_evaluations": [
                {
                    "stat_name": e.stat_name,
                    "n_samples": e.n_samples,
                    "rmse": e.rmse,
                    "weighted_rmse": e.weighted_rmse,
                    "mae": e.mae,
                    "spearman_rho": e.spearman_rho,
                }
                for e in self.stat_evaluations
            ],
        }
        if self.baseline_comparisons is not None:
            result["baseline_comparisons"] = [
                {
                    "stat_name": c.stat_name,
                    "model_rmse": c.model_rmse,
                    "baseline_rmse": c.baseline_rmse,
                    "rmse_improvement": c.rmse_improvement,
                    "rmse_improvement_pct": c.rmse_improvement_pct,
                    "model_spearman": c.model_spearman,
                    "baseline_spearman": c.baseline_spearman,
                }
                for c in self.baseline_comparisons
            ]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> HoldoutEvaluationReport:
        """Deserialize from a dictionary."""
        stat_evaluations = tuple(
            HoldoutEvaluation(
                stat_name=e["stat_name"],
                n_samples=e["n_samples"],
                rmse=e["rmse"],
                weighted_rmse=e["weighted_rmse"],
                mae=e["mae"],
                spearman_rho=e["spearman_rho"],
            )
            for e in data["stat_evaluations"]
        )
        baseline_comparisons = None
        if data.get("baseline_comparisons"):
            baseline_comparisons = tuple(
                BaselineComparison(
                    stat_name=c["stat_name"],
                    model_rmse=c["model_rmse"],
                    baseline_rmse=c["baseline_rmse"],
                    rmse_improvement=c["rmse_improvement"],
                    rmse_improvement_pct=c["rmse_improvement_pct"],
                    model_spearman=c["model_spearman"],
                    baseline_spearman=c["baseline_spearman"],
                )
                for c in data["baseline_comparisons"]
            )
        return cls(
            model_name=data["model_name"],
            test_years=tuple(data["test_years"]),
            n_samples=data["n_samples"],
            stat_evaluations=stat_evaluations,
            baseline_comparisons=baseline_comparisons,
        )


def compute_weighted_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute weighted RMSE.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        weights: Sample weights (e.g., PA for baseball stats).

    Returns:
        Weighted RMSE value.

    Raises:
        ValueError: If arrays are empty or have mismatched lengths.
    """
    import numpy as np

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    if len(y_true) != len(y_pred) or len(y_true) != len(weights):
        raise ValueError(
            f"Array length mismatch: y_true={len(y_true)}, " f"y_pred={len(y_pred)}, weights={len(weights)}"
        )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    # Normalize weights
    weight_sum = weights.sum()
    if weight_sum == 0:
        return 0.0

    normalized_weights = weights / weight_sum

    # Weighted MSE
    squared_errors = (y_true - y_pred) ** 2
    weighted_mse = float(np.sum(normalized_weights * squared_errors))

    return float(np.sqrt(weighted_mse))


def compute_spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Spearman rho correlation coefficient.

    Raises:
        ValueError: If arrays are empty or have mismatched lengths.
    """
    import numpy as np
    from scipy.stats import spearmanr

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if len(y_true) < 2:
        return 0.0

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = spearmanr(y_true, y_pred)
    # Handle NaN (can occur with constant arrays)
    # scipy.stats returns SignificanceResult with 'statistic' attribute
    correlation = result.statistic
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


@dataclass
class HoldoutEvaluator:
    """Generic evaluator for ML models on held-out test data.

    This evaluator computes standard metrics (RMSE, weighted RMSE, MAE,
    Spearman rho) for multi-output models where predictions and targets
    are dictionaries mapping stat names to arrays.

    Example:
        evaluator = HoldoutEvaluator(model_name="mle_v1", test_years=(2024,))
        report = evaluator.evaluate(
            y_true={"hr": hr_actual, "so": so_actual},
            y_pred={"hr": hr_pred, "so": so_pred},
            sample_weights=mlb_pa,
        )
    """

    model_name: str
    test_years: tuple[int, ...]

    def evaluate(
        self,
        y_true: dict[str, np.ndarray],
        y_pred: dict[str, np.ndarray],
        sample_weights: np.ndarray | None = None,
    ) -> HoldoutEvaluationReport:
        """Evaluate model predictions against actual values.

        Args:
            y_true: Dict mapping stat names to arrays of actual values.
            y_pred: Dict mapping stat names to arrays of predicted values.
            sample_weights: Optional array of sample weights for weighted RMSE.

        Returns:
            HoldoutEvaluationReport with per-stat metrics.

        Raises:
            ValueError: If stat names don't match or arrays are empty.
        """
        import numpy as np

        if set(y_true.keys()) != set(y_pred.keys()):
            raise ValueError(
                f"Stat names don't match: y_true has {set(y_true.keys())}, " f"y_pred has {set(y_pred.keys())}"
            )

        if not y_true:
            raise ValueError("No stats provided for evaluation")

        # Get sample count from first stat
        first_stat = next(iter(y_true.keys()))
        n_samples = len(y_true[first_stat])

        if n_samples == 0:
            raise ValueError("Cannot evaluate on empty arrays")

        # Default weights to uniform if not provided
        if sample_weights is None:
            sample_weights = np.ones(n_samples)

        stat_evaluations: list[HoldoutEvaluation] = []

        for stat_name in sorted(y_true.keys()):
            true_vals = np.asarray(y_true[stat_name])
            pred_vals = np.asarray(y_pred[stat_name])

            if len(true_vals) != n_samples or len(pred_vals) != n_samples:
                raise ValueError(
                    f"Array length mismatch for stat {stat_name}: "
                    f"expected {n_samples}, got true={len(true_vals)}, pred={len(pred_vals)}"
                )

            # Compute metrics
            rmse = float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))
            weighted_rmse = compute_weighted_rmse(true_vals, pred_vals, sample_weights)
            mae = float(np.mean(np.abs(true_vals - pred_vals)))
            spearman = compute_spearman_rho(true_vals, pred_vals)

            stat_evaluations.append(
                HoldoutEvaluation(
                    stat_name=stat_name,
                    n_samples=n_samples,
                    rmse=rmse,
                    weighted_rmse=weighted_rmse,
                    mae=mae,
                    spearman_rho=spearman,
                )
            )

        return HoldoutEvaluationReport(
            model_name=self.model_name,
            test_years=self.test_years,
            n_samples=n_samples,
            stat_evaluations=tuple(stat_evaluations),
        )

    def compare_to_baseline(
        self,
        y_true: dict[str, np.ndarray],
        y_pred: dict[str, np.ndarray],
        y_baseline: dict[str, np.ndarray],
        sample_weights: np.ndarray | None = None,
        baseline_name: str = "baseline",
    ) -> HoldoutEvaluationReport:
        """Evaluate model and compare to a baseline.

        Args:
            y_true: Dict mapping stat names to arrays of actual values.
            y_pred: Dict mapping stat names to arrays of model predictions.
            y_baseline: Dict mapping stat names to arrays of baseline predictions.
            sample_weights: Optional array of sample weights for weighted RMSE.
            baseline_name: Name for the baseline in the report.

        Returns:
            HoldoutEvaluationReport with per-stat metrics and baseline comparisons.

        Raises:
            ValueError: If stat names don't match or arrays are empty.
        """
        import numpy as np

        # First get the base evaluation
        report = self.evaluate(y_true, y_pred, sample_weights)

        # Validate baseline has same stats
        if set(y_true.keys()) != set(y_baseline.keys()):
            raise ValueError(
                f"Baseline stat names don't match: y_true has {set(y_true.keys())}, "
                f"y_baseline has {set(y_baseline.keys())}"
            )

        n_samples = report.n_samples
        comparisons: list[BaselineComparison] = []

        for eval_result in report.stat_evaluations:
            stat_name = eval_result.stat_name
            true_vals = np.asarray(y_true[stat_name])
            baseline_vals = np.asarray(y_baseline[stat_name])

            if len(baseline_vals) != n_samples:
                raise ValueError(
                    f"Baseline array length mismatch for stat {stat_name}: "
                    f"expected {n_samples}, got {len(baseline_vals)}"
                )

            # Compute baseline metrics
            baseline_rmse = float(np.sqrt(np.mean((true_vals - baseline_vals) ** 2)))
            baseline_spearman = compute_spearman_rho(true_vals, baseline_vals)

            # Compute improvement
            rmse_improvement = baseline_rmse - eval_result.rmse
            rmse_improvement_pct = (rmse_improvement / baseline_rmse * 100) if baseline_rmse > 0 else 0.0

            comparisons.append(
                BaselineComparison(
                    stat_name=stat_name,
                    model_rmse=eval_result.rmse,
                    baseline_rmse=baseline_rmse,
                    rmse_improvement=rmse_improvement,
                    rmse_improvement_pct=rmse_improvement_pct,
                    model_spearman=eval_result.spearman_rho,
                    baseline_spearman=baseline_spearman,
                )
            )

        return HoldoutEvaluationReport(
            model_name=f"{self.model_name} vs {baseline_name}",
            test_years=report.test_years,
            n_samples=report.n_samples,
            stat_evaluations=report.stat_evaluations,
            baseline_comparisons=tuple(comparisons),
        )
