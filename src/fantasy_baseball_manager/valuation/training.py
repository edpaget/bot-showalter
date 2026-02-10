"""Training pipeline for ridge regression valuation models.

Orchestrates data loading, splitting, training, and evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fantasy_baseball_manager.ml.validation import (
    compute_spearman_rho,
    compute_validation_metrics,
)
from fantasy_baseball_manager.valuation.features import (
    BATTER_FEATURE_NAMES,
    PITCHER_FEATURE_NAMES,
    batter_training_rows_to_arrays,
    pitcher_training_rows_to_arrays,
)
from fantasy_baseball_manager.valuation.ridge_model import (
    RidgeValuationConfig,
    RidgeValuationModel,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.adp.training_dataset import (
        BatterTrainingRow,
        PitcherTrainingRow,
    )
    from fantasy_baseball_manager.projections.models import ProjectionSystem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValuationEvaluation:
    """Evaluation results for a trained valuation model."""

    training_years: tuple[int, ...]
    test_years: tuple[int, ...]
    n_train: int
    n_test: int
    spearman_rho: float
    rmse: float
    mae: float
    top_50_precision: float
    coefficient_analysis: dict[str, float]


@dataclass(frozen=True)
class ValuationTrainingResult:
    """Combined result of training batter + pitcher models."""

    batter_model: RidgeValuationModel
    pitcher_model: RidgeValuationModel
    batter_eval: ValuationEvaluation
    pitcher_eval: ValuationEvaluation


def _compute_top_k_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
) -> float:
    """Fraction of true top-k players (lowest ADP) in predicted top-k.

    Both arrays are in log-ADP space, so lower = more valuable.
    """
    if len(y_true) < k:
        k = len(y_true)
    if k == 0:
        return 0.0
    true_top_k = set(np.argsort(y_true)[:k])
    pred_top_k = set(np.argsort(y_pred)[:k])
    return len(true_top_k & pred_top_k) / k


def train_ridge_valuation(
    training_rows: list[BatterTrainingRow] | list[PitcherTrainingRow],
    test_rows: list[BatterTrainingRow] | list[PitcherTrainingRow],
    player_type: str,
    config: RidgeValuationConfig | None = None,
) -> tuple[RidgeValuationModel, ValuationEvaluation]:
    """Train ridge model on training rows, evaluate on test rows.

    Args:
        training_rows: Rows for the training split.
        test_rows: Rows for the test split.
        player_type: "batter" or "pitcher".
        config: Optional config overrides.

    Returns:
        Tuple of (fitted model, evaluation metrics).
    """
    config = config or RidgeValuationConfig()

    if player_type == "batter":
        X_train, y_train = batter_training_rows_to_arrays(training_rows)  # type: ignore[arg-type]
        X_test, y_test = batter_training_rows_to_arrays(test_rows)  # type: ignore[arg-type]
        feature_names = BATTER_FEATURE_NAMES
    else:
        X_train, y_train = pitcher_training_rows_to_arrays(training_rows)  # type: ignore[arg-type]
        X_test, y_test = pitcher_training_rows_to_arrays(test_rows)  # type: ignore[arg-type]
        feature_names = PITCHER_FEATURE_NAMES

    training_years = tuple(sorted({r.year for r in training_rows}))
    test_years = tuple(sorted({r.year for r in test_rows}))

    model = RidgeValuationModel(player_type=player_type, config=config)
    model.fit(X_train, y_train, feature_names)
    model.training_years = training_years

    # Evaluate on test set
    y_pred = model.predict(X_test)
    metrics = compute_validation_metrics(y_test, y_pred, "test")
    spearman = compute_spearman_rho(y_test, y_pred)
    top_50 = _compute_top_k_precision(y_test, y_pred, k=50)

    evaluation = ValuationEvaluation(
        training_years=training_years,
        test_years=test_years,
        n_train=len(X_train),
        n_test=len(X_test),
        spearman_rho=spearman,
        rmse=metrics.rmse,
        mae=metrics.mae,
        top_50_precision=top_50,
        coefficient_analysis=model.coefficients(),
    )

    model.validation_metrics = {
        "spearman_rho": spearman,
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "top_50_precision": top_50,
    }

    return model, evaluation


def train_multi_year(
    years: list[int],
    test_years: list[int],
    system: ProjectionSystem,
    config: RidgeValuationConfig | None = None,
) -> ValuationTrainingResult:
    """Full training pipeline: load data, split by year, train, evaluate.

    Args:
        years: All years to include (training + test).
        test_years: Years to hold out for evaluation.
        system: Projection system (steamer, zips, etc.).
        config: Optional config overrides.

    Returns:
        ValuationTrainingResult with both batter and pitcher models.
    """
    from fantasy_baseball_manager.adp.csv_resolver import ADPCSVResolver
    from fantasy_baseball_manager.adp.training_dataset import build_multi_year_dataset
    from fantasy_baseball_manager.projections.csv_resolver import CSVProjectionResolver

    proj_resolver = CSVProjectionResolver()
    adp_resolver = ADPCSVResolver()

    effective_config = config or RidgeValuationConfig()

    all_batters, all_pitchers, _per_year = build_multi_year_dataset(
        years,
        proj_resolver,
        adp_resolver,
        system,
        min_pa=effective_config.min_pa,
        min_ip=effective_config.min_ip,
    )

    test_year_set = set(test_years)
    batter_train = [r for r in all_batters if r.year not in test_year_set]
    batter_test = [r for r in all_batters if r.year in test_year_set]
    pitcher_train = [r for r in all_pitchers if r.year not in test_year_set]
    pitcher_test = [r for r in all_pitchers if r.year in test_year_set]

    logger.info(
        "Split: %d/%d batter train/test, %d/%d pitcher train/test",
        len(batter_train),
        len(batter_test),
        len(pitcher_train),
        len(pitcher_test),
    )

    batter_model, batter_eval = train_ridge_valuation(
        batter_train, batter_test, "batter", effective_config
    )
    pitcher_model, pitcher_eval = train_ridge_valuation(
        pitcher_train, pitcher_test, "pitcher", effective_config
    )

    return ValuationTrainingResult(
        batter_model=batter_model,
        pitcher_model=pitcher_model,
        batter_eval=batter_eval,
        pitcher_eval=pitcher_eval,
    )
