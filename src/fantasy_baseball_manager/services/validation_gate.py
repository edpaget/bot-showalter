"""Validation gate — pre-flight confidence estimator and full validation orchestrator.

Analyzes per-fold CV results to estimate whether a candidate model
change is likely to pass the full comparison protocol, and orchestrates
the complete train → predict → compare sequence for multiple holdout seasons.
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from fantasy_baseball_manager.domain import (
    ComparisonResult,
    ModelConfig,
    PredictResult,
    Projection,
    StatDistribution,
    check_regression,
    summarize_comparison,
)
from fantasy_baseball_manager.models.gbm_training import (
    extract_features,
    extract_targets,
    fit_models,
    score_predictions,
)
from fantasy_baseball_manager.models.sampling import temporal_expanding_cv

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import RegressionCheckResult
    from fantasy_baseball_manager.models import Model


@dataclass(frozen=True)
class PreflightThresholds:
    """Configurable thresholds for pre-flight confidence classification."""

    high_win_rate: float = 0.75
    high_target_pct: float = 0.80
    medium_win_rate: float = 0.60
    medium_target_pct: float = 0.60


@dataclass(frozen=True)
class TargetPreflightDetail:
    """Per-target pre-flight metrics across CV folds."""

    target: str
    win_rate: float
    mean_delta: float
    delta_std: float


@dataclass(frozen=True)
class PreflightResult:
    """Overall pre-flight verdict with per-target breakdown."""

    details: tuple[TargetPreflightDetail, ...]
    confidence: str
    recommendation: str


def preflight_check(
    cv_results: list[dict[str, float]],
    baseline_cv_results: list[dict[str, float]],
    thresholds: PreflightThresholds | None = None,
) -> PreflightResult:
    """Analyze per-fold CV results and estimate confidence that the full gate will pass.

    Args:
        cv_results: Per-fold dicts mapping target name → RMSE for the candidate.
        baseline_cv_results: Per-fold dicts mapping target name → RMSE for the baseline.
        thresholds: Optional custom thresholds for confidence classification.

    Returns:
        PreflightResult with per-target details, confidence level, and recommendation.
    """
    if thresholds is None:
        thresholds = PreflightThresholds()

    n_folds = len(cv_results)
    if n_folds == 0:
        return PreflightResult(details=(), confidence="low", recommendation="skip")

    targets = sorted(cv_results[0].keys())
    details: list[TargetPreflightDetail] = []

    for target in targets:
        wins = sum(1 for i in range(n_folds) if cv_results[i][target] < baseline_cv_results[i][target])
        win_rate = wins / n_folds
        deltas = [cv_results[i][target] - baseline_cv_results[i][target] for i in range(n_folds)]
        mean_delta = statistics.mean(deltas)
        delta_std = statistics.stdev(deltas) if n_folds > 1 else 0.0
        details.append(
            TargetPreflightDetail(
                target=target,
                win_rate=win_rate,
                mean_delta=mean_delta,
                delta_std=delta_std,
            )
        )

    n_targets = len(details)
    if n_targets == 0:
        return PreflightResult(details=(), confidence="low", recommendation="skip")

    high_count = sum(1 for d in details if d.win_rate >= thresholds.high_win_rate)
    medium_count = sum(1 for d in details if d.win_rate >= thresholds.medium_win_rate)

    if high_count / n_targets >= thresholds.high_target_pct:
        confidence = "high"
    elif medium_count / n_targets >= thresholds.medium_target_pct:
        confidence = "medium"
    else:
        confidence = "low"

    recommendation_map = {"high": "proceed", "medium": "marginal", "low": "skip"}
    recommendation = recommendation_map[confidence]

    return PreflightResult(
        details=tuple(details),
        confidence=confidence,
        recommendation=recommendation,
    )


def score_cv_folds(
    columns: list[str],
    targets: list[str],
    rows_by_season: dict[int, list[dict[str, Any]]],
    seasons: list[int],
    params: dict[str, Any],
) -> list[dict[str, float]]:
    """Score a feature set across temporal expanding CV folds.

    Builds temporal expanding CV folds, trains a GBM on each fold's training
    data, and records per-target RMSE on the holdout fold.

    Args:
        columns: Feature column names to use.
        targets: Target names (without ``target_`` prefix).
        rows_by_season: Training data grouped by season.
        seasons: Season years to use for CV splits.
        params: GBM hyperparameters.

    Returns:
        List of per-fold dicts mapping target name → RMSE.
    """
    cv_splits = list(temporal_expanding_cv(seasons))
    fold_results: list[dict[str, float]] = []

    for train_seasons, test_season in cv_splits:
        train_rows = [row for s in train_seasons for row in rows_by_season.get(s, [])]
        test_rows = rows_by_season.get(test_season, [])

        X_train = extract_features(train_rows, columns)
        y_train = extract_targets(train_rows, targets)
        X_test = extract_features(test_rows, columns)
        y_test = extract_targets(test_rows, targets)

        models = fit_models(X_train, y_train, params)
        metrics = score_predictions(models, X_test, y_test)

        fold_dict: dict[str, float] = {}
        for key, value in metrics.items():
            target_name = key.removeprefix("rmse_")
            fold_dict[target_name] = value
        fold_results.append(fold_dict)

    return fold_results


# ---------------------------------------------------------------------------
# Full validation orchestrator
# ---------------------------------------------------------------------------


class _Evaluator(Protocol):
    def compare(
        self,
        systems: list[tuple[str, str]],
        season: int,
        *,
        top: int | None = ...,
    ) -> ComparisonResult: ...


class _ProjectionRepo(Protocol):
    def upsert(self, projection: Projection) -> int: ...
    def get_by_system_version(self, system: str, version: str) -> list[Any]: ...
    def delete_by_system_version(self, system: str, version: str) -> int: ...
    def upsert_distributions(self, projection_id: int, distributions: list[Any]) -> None: ...


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FullValidationConfig:
    model_name: str
    old_version: str
    new_version: str
    old_params: dict[str, Any]
    new_params: dict[str, Any]
    holdout_seasons: list[int]
    train_seasons: list[int]
    top: int | None
    data_dir: str
    artifacts_dir: str


@dataclass(frozen=True)
class ValidationSegmentResult:
    season: int
    segment: str
    check: RegressionCheckResult


@dataclass(frozen=True)
class ValidationResult:
    model_name: str
    old_version: str
    new_version: str
    segments: list[ValidationSegmentResult] = field(default_factory=list)
    preflight: PreflightResult | None = None

    @property
    def passed(self) -> bool:
        return all(s.check.passed for s in self.segments)


class FullValidationRunner:
    def __init__(
        self,
        model: Model,
        evaluator: _Evaluator,
        projection_repo: _ProjectionRepo,
    ) -> None:
        self._model = model
        self._evaluator = evaluator
        self._projection_repo = projection_repo

    def run(
        self,
        config: FullValidationConfig,
        preflight: PreflightResult | None = None,
    ) -> ValidationResult:
        segments: list[ValidationSegmentResult] = []

        for holdout in config.holdout_seasons:
            effective_train = [s for s in config.train_seasons if s != holdout]
            old_tag = f"{config.old_version}-h{holdout}"
            new_tag = f"{config.new_version}-h{holdout}"

            # Train + predict old version (skip if predictions exist)
            self._train_and_predict(
                config=config,
                version_tag=old_tag,
                params=config.old_params,
                train_seasons=effective_train,
                predict_season=holdout,
            )

            # Train + predict new version (skip if predictions exist)
            self._train_and_predict(
                config=config,
                version_tag=new_tag,
                params=config.new_params,
                train_seasons=effective_train,
                predict_season=holdout,
            )

            # Full-population check
            comparison = self._evaluator.compare(
                systems=[
                    (config.model_name, old_tag),
                    (config.model_name, new_tag),
                ],
                season=holdout,
            )
            summary = summarize_comparison(comparison)
            check = check_regression(summary)
            segments.append(ValidationSegmentResult(season=holdout, segment="full", check=check))
            logger.info("Holdout %d full: %s", holdout, check.explanation)

            # Top-N check
            if config.top is not None:
                comparison_top = self._evaluator.compare(
                    systems=[
                        (config.model_name, old_tag),
                        (config.model_name, new_tag),
                    ],
                    season=holdout,
                    top=config.top,
                )
                summary_top = summarize_comparison(comparison_top)
                check_top = check_regression(summary_top)
                segments.append(
                    ValidationSegmentResult(
                        season=holdout,
                        segment=f"top-{config.top}",
                        check=check_top,
                    )
                )
                logger.info("Holdout %d top-%d: %s", holdout, config.top, check_top.explanation)

        return ValidationResult(
            model_name=config.model_name,
            old_version=config.old_version,
            new_version=config.new_version,
            segments=segments,
            preflight=preflight,
        )

    def cleanup(self, config: FullValidationConfig) -> None:
        for holdout in config.holdout_seasons:
            for version in (config.old_version, config.new_version):
                tag = f"{version}-h{holdout}"
                count = self._projection_repo.delete_by_system_version(config.model_name, tag)
                logger.info("Cleaned up %d predictions for %s/%s", count, config.model_name, tag)

    def _train_and_predict(
        self,
        *,
        config: FullValidationConfig,
        version_tag: str,
        params: dict[str, Any],
        train_seasons: list[int],
        predict_season: int,
    ) -> None:
        # Reuse check: skip train+predict if predictions already exist
        existing = self._projection_repo.get_by_system_version(config.model_name, version_tag)
        if existing:
            logger.info("Reusing existing predictions for %s/%s", config.model_name, version_tag)
            return

        train_config = ModelConfig(
            data_dir=config.data_dir,
            artifacts_dir=config.artifacts_dir,
            seasons=train_seasons,
            model_params=params,
            version=version_tag,
        )
        logger.info("Training %s/%s", config.model_name, version_tag)
        self._model.train(train_config)  # type: ignore[union-attr]

        predict_config = ModelConfig(
            data_dir=config.data_dir,
            artifacts_dir=config.artifacts_dir,
            seasons=[predict_season],
            model_params=params,
            version=version_tag,
        )
        logger.info("Predicting %s/%s for %d", config.model_name, version_tag, predict_season)
        predict_result: PredictResult = self._model.predict(predict_config)  # type: ignore[union-attr]

        self._persist_predictions(predict_result, config.model_name, version_tag)

    def _persist_predictions(self, result: PredictResult, model_name: str, version: str) -> None:
        for pred in result.predictions:
            if "player_id" not in pred or "season" not in pred:
                continue
            stat_json = {k: v for k, v in pred.items() if k not in ("player_id", "season", "player_type")}
            proj = Projection(
                player_id=pred["player_id"],
                season=pred["season"],
                system=model_name,
                version=version,
                player_type=pred.get("player_type", "batter"),
                stat_json=stat_json,
            )
            proj_id = self._projection_repo.upsert(proj)

            if result.distributions is not None:
                player_dists = [
                    d
                    for d in result.distributions
                    if d["player_id"] == pred["player_id"] and d["player_type"] == pred.get("player_type", "batter")
                ]
                if player_dists:
                    stat_dists = [
                        StatDistribution(
                            stat=d["stat"],
                            p10=d["p10"],
                            p25=d["p25"],
                            p50=d["p50"],
                            p75=d["p75"],
                            p90=d["p90"],
                            mean=d["mean"],
                            std=d["std"],
                        )
                        for d in player_dists
                    ]
                    self._projection_repo.upsert_distributions(proj_id, stat_dists)
