from pathlib import Path
from typing import Any

from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import AnyFeature
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    Evaluator,
    ModelConfig,
    PredictResult,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.models.statcast_gbm.features import (
    batter_feature_columns,
    build_batter_feature_set,
    build_batter_training_set,
    build_pitcher_feature_set,
)
from fantasy_baseball_manager.models.statcast_gbm.serialization import load_models, save_models
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS
from fantasy_baseball_manager.models.statcast_gbm.training import (
    extract_features,
    extract_targets,
    fit_models,
    score_predictions,
)


@register("statcast-gbm")
class StatcastGBMModel:
    def __init__(
        self,
        assembler: DatasetAssembler | None = None,
        evaluator: Evaluator | None = None,
    ) -> None:
        self._assembler = assembler
        self._evaluator = evaluator

    @property
    def name(self) -> str:
        return "statcast-gbm"

    @property
    def description(self) -> str:
        return "Gradient-boosted model using Statcast features for player projections."

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "evaluate", "predict", "ablate"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.FILE.value

    @property
    def declared_features(self) -> tuple[AnyFeature, ...]:
        batter_fs = build_batter_feature_set([])
        pitcher_fs = build_pitcher_feature_set([])
        return batter_fs.features + pitcher_fs.features

    def prepare(self, config: ModelConfig) -> PrepareResult:
        assert self._assembler is not None, "assembler is required for prepare"
        batter_fs = build_batter_feature_set(config.seasons)
        pitcher_fs = build_pitcher_feature_set(config.seasons)

        bat_handle = self._assembler.get_or_materialize(batter_fs)
        pitch_handle = self._assembler.get_or_materialize(pitcher_fs)

        return PrepareResult(
            model_name=self.name,
            rows_processed=bat_handle.row_count + pitch_handle.row_count,
            artifacts_path=config.artifacts_dir,
        )

    def train(self, config: ModelConfig) -> TrainResult:
        assert self._assembler is not None, "assembler is required for train"

        train_fs = build_batter_training_set(config.seasons)
        handle = self._assembler.get_or_materialize(train_fs)

        train_seasons = config.seasons[:-1]
        holdout_seasons = [config.seasons[-1]]
        splits = self._assembler.split(handle, train=train_seasons, holdout=holdout_seasons)

        train_rows = self._assembler.read(splits.train)
        feature_columns = batter_feature_columns()
        targets = list(BATTER_TARGETS)

        X_train = extract_features(train_rows, feature_columns)
        y_train = extract_targets(train_rows, targets)

        models = fit_models(X_train, y_train, config.model_params)

        metrics: dict[str, float] = {}
        if splits.holdout is not None:
            holdout_rows = self._assembler.read(splits.holdout)
            X_holdout = extract_features(holdout_rows, feature_columns)
            y_holdout = extract_targets(holdout_rows, targets)
            metrics = score_predictions(models, X_holdout, y_holdout)

        artifact_path = self._artifact_path(config)
        artifact_path.mkdir(parents=True, exist_ok=True)
        save_models(models, artifact_path / "batter_models.joblib")

        return TrainResult(
            model_name=self.name,
            metrics=metrics,
            artifacts_path=str(artifact_path),
        )

    def evaluate(self, config: ModelConfig) -> SystemMetrics:
        assert self._evaluator is not None, "evaluator is required for evaluate"
        version = config.version or "latest"
        season = config.seasons[0]
        return self._evaluator.evaluate(self.name, version, season)

    def predict(self, config: ModelConfig) -> PredictResult:
        assert self._assembler is not None, "assembler is required for predict"

        predict_fs = build_batter_feature_set(config.seasons)
        handle = self._assembler.get_or_materialize(predict_fs)
        rows = self._assembler.read(handle)

        artifact_path = self._artifact_path(config)
        models = load_models(artifact_path / "batter_models.joblib")

        feature_columns = batter_feature_columns()
        X = extract_features(rows, feature_columns)

        predictions_by_row: list[dict[str, Any]] = []
        target_preds: dict[str, list[float]] = {}
        for target_name, model in models.items():
            target_preds[target_name] = list(model.predict(X))

        for i, row in enumerate(rows):
            pred: dict[str, Any] = {
                "player_id": row["player_id"],
                "season": row["season"],
                "player_type": "batter",
            }
            for target_name in models:
                pred[target_name] = target_preds[target_name][i]
            predictions_by_row.append(pred)

        return PredictResult(
            model_name=self.name,
            predictions=predictions_by_row,
            output_path=config.output_dir or config.artifacts_dir,
        )

    def ablate(self, config: ModelConfig) -> AblationResult:
        return AblationResult(
            model_name=self.name,
            feature_impacts={},
        )

    def _artifact_path(self, config: ModelConfig) -> Path:
        return Path(config.artifacts_dir) / self.name / (config.version or "latest")
