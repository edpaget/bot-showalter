import logging
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

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import RegressionCheckResult
    from fantasy_baseball_manager.models import Model
    from fantasy_baseball_manager.repos import ProjectionRepo


class _Evaluator(Protocol):
    def compare(
        self,
        systems: list[tuple[str, str]],
        season: int,
        *,
        top: int | None = ...,
    ) -> ComparisonResult: ...


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateConfig:
    model_name: str
    base_training_seasons: list[int]
    holdout_seasons: list[int]
    baseline_system: str
    baseline_version: str
    top: int | None
    model_params: dict[str, Any]
    data_dir: str
    artifacts_dir: str


@dataclass(frozen=True)
class GateSegmentResult:
    season: int
    segment: str
    check: RegressionCheckResult


@dataclass(frozen=True)
class GateResult:
    model_name: str
    baseline: str
    segments: list[GateSegmentResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(s.check.passed for s in self.segments)


class RegressionGateRunner:
    def __init__(
        self,
        model: Model,
        evaluator: _Evaluator,
        projection_repo: ProjectionRepo,
    ) -> None:
        self._model = model
        self._evaluator = evaluator
        self._projection_repo = projection_repo

    def run(self, config: GateConfig) -> GateResult:
        segments: list[GateSegmentResult] = []
        baseline_label = f"{config.baseline_system}/{config.baseline_version}"

        for holdout in config.holdout_seasons:
            version = f"gate-h{holdout}"
            train_seasons = [*config.base_training_seasons, holdout]

            train_config = ModelConfig(
                data_dir=config.data_dir,
                artifacts_dir=config.artifacts_dir,
                seasons=train_seasons,
                model_params=config.model_params,
                version=version,
            )
            logger.info("Training %s for holdout %d", config.model_name, holdout)
            self._model.train(train_config)  # type: ignore[union-attr]

            predict_config = ModelConfig(
                data_dir=config.data_dir,
                artifacts_dir=config.artifacts_dir,
                seasons=[holdout],
                model_params=config.model_params,
                version=version,
            )
            logger.info("Predicting %s for holdout %d", config.model_name, holdout)
            predict_result: PredictResult = self._model.predict(predict_config)  # type: ignore[union-attr]

            self._persist_predictions(predict_result, config.model_name, version)

            # Full-population check
            comparison = self._evaluator.compare(
                systems=[
                    (config.baseline_system, config.baseline_version),
                    (config.model_name, version),
                ],
                season=holdout,
            )
            summary = summarize_comparison(comparison)
            check = check_regression(summary)
            segments.append(GateSegmentResult(season=holdout, segment="full", check=check))
            logger.info("Holdout %d full: %s", holdout, check.explanation)

            # Top-N check
            if config.top is not None:
                comparison_top = self._evaluator.compare(
                    systems=[
                        (config.baseline_system, config.baseline_version),
                        (config.model_name, version),
                    ],
                    season=holdout,
                    top=config.top,
                )
                summary_top = summarize_comparison(comparison_top)
                check_top = check_regression(summary_top)
                segments.append(
                    GateSegmentResult(
                        season=holdout,
                        segment=f"top-{config.top}",
                        check=check_top,
                    )
                )
                logger.info("Holdout %d top-%d: %s", holdout, config.top, check_top.explanation)

        return GateResult(
            model_name=config.model_name,
            baseline=baseline_label,
            segments=segments,
        )

    def cleanup(self, config: GateConfig) -> None:
        for holdout in config.holdout_seasons:
            version = f"gate-h{holdout}"
            count = self._projection_repo.delete_by_system_version(config.model_name, version)
            logger.info("Cleaned up %d predictions for %s/%s", count, config.model_name, version)

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
