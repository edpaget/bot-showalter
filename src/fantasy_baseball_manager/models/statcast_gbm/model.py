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
    batter_preseason_feature_columns,
    build_batter_feature_set,
    build_batter_preseason_set,
    build_batter_preseason_training_set,
    build_batter_training_set,
    build_pitcher_feature_set,
    build_pitcher_preseason_set,
    build_pitcher_preseason_training_set,
    build_pitcher_training_set,
    pitcher_feature_columns,
    pitcher_preseason_feature_columns,
)
from fantasy_baseball_manager.models.statcast_gbm.serialization import load_models, save_models
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS
from fantasy_baseball_manager.models.statcast_gbm.training import (
    compute_permutation_importance,
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

    def _is_preseason(self, config: ModelConfig) -> bool:
        return config.model_params.get("mode") == "preseason"

    def prepare(self, config: ModelConfig) -> PrepareResult:
        assert self._assembler is not None, "assembler is required for prepare"
        preseason = self._is_preseason(config)
        batter_fs = (
            build_batter_preseason_set(config.seasons) if preseason else build_batter_feature_set(config.seasons)
        )
        pitcher_fs = (
            build_pitcher_preseason_set(config.seasons) if preseason else build_pitcher_feature_set(config.seasons)
        )

        bat_handle = self._assembler.get_or_materialize(batter_fs)
        pitch_handle = self._assembler.get_or_materialize(pitcher_fs)

        return PrepareResult(
            model_name=self.name,
            rows_processed=bat_handle.row_count + pitch_handle.row_count,
            artifacts_path=config.artifacts_dir,
        )

    def train(self, config: ModelConfig) -> TrainResult:
        assert self._assembler is not None, "assembler is required for train"
        if len(config.seasons) < 2:
            msg = f"train requires at least 2 seasons (got {len(config.seasons)})"
            raise ValueError(msg)

        preseason = self._is_preseason(config)
        train_seasons = config.seasons[:-1]
        holdout_seasons = [config.seasons[-1]]
        metrics: dict[str, float] = {}
        artifact_path = self._artifact_path(config)
        if preseason:
            artifact_path = artifact_path / "preseason"
        artifact_path.mkdir(parents=True, exist_ok=True)

        # --- Batter training ---
        bat_fs = (
            build_batter_preseason_training_set(config.seasons)
            if preseason
            else build_batter_training_set(config.seasons)
        )
        bat_handle = self._assembler.get_or_materialize(bat_fs)
        bat_splits = self._assembler.split(bat_handle, train=train_seasons, holdout=holdout_seasons)

        bat_train_rows = self._assembler.read(bat_splits.train)
        bat_feature_cols = batter_preseason_feature_columns() if preseason else batter_feature_columns()
        bat_targets = list(BATTER_TARGETS)

        bat_X_train = extract_features(bat_train_rows, bat_feature_cols)
        bat_y_train = extract_targets(bat_train_rows, bat_targets)
        bat_models = fit_models(bat_X_train, bat_y_train, config.model_params)

        if bat_splits.holdout is not None:
            bat_holdout_rows = self._assembler.read(bat_splits.holdout)
            bat_X_holdout = extract_features(bat_holdout_rows, bat_feature_cols)
            bat_y_holdout = extract_targets(bat_holdout_rows, bat_targets)
            bat_metrics = score_predictions(bat_models, bat_X_holdout, bat_y_holdout)
            for key, value in bat_metrics.items():
                metrics[f"batter_{key}"] = value

        save_models(bat_models, artifact_path / "batter_models.joblib")

        # --- Pitcher training ---
        pit_fs = (
            build_pitcher_preseason_training_set(config.seasons)
            if preseason
            else build_pitcher_training_set(config.seasons)
        )
        pit_handle = self._assembler.get_or_materialize(pit_fs)
        pit_splits = self._assembler.split(pit_handle, train=train_seasons, holdout=holdout_seasons)

        pit_train_rows = self._assembler.read(pit_splits.train)
        pit_feature_cols = pitcher_preseason_feature_columns() if preseason else pitcher_feature_columns()
        pit_targets = list(PITCHER_TARGETS)

        pit_X_train = extract_features(pit_train_rows, pit_feature_cols)
        pit_y_train = extract_targets(pit_train_rows, pit_targets)
        pit_models = fit_models(pit_X_train, pit_y_train, config.model_params)

        if pit_splits.holdout is not None:
            pit_holdout_rows = self._assembler.read(pit_splits.holdout)
            pit_X_holdout = extract_features(pit_holdout_rows, pit_feature_cols)
            pit_y_holdout = extract_targets(pit_holdout_rows, pit_targets)
            pit_metrics = score_predictions(pit_models, pit_X_holdout, pit_y_holdout)
            for key, value in pit_metrics.items():
                metrics[f"pitcher_{key}"] = value

        save_models(pit_models, artifact_path / "pitcher_models.joblib")

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

        preseason = self._is_preseason(config)
        artifact_path = self._artifact_path(config)
        if preseason:
            artifact_path = artifact_path / "preseason"
        predictions_by_row: list[dict[str, Any]] = []

        # --- Batter predictions ---
        bat_fs = build_batter_preseason_set(config.seasons) if preseason else build_batter_feature_set(config.seasons)
        bat_handle = self._assembler.get_or_materialize(bat_fs)
        bat_rows = self._assembler.read(bat_handle)
        bat_models = load_models(artifact_path / "batter_models.joblib")
        bat_feature_cols = batter_preseason_feature_columns() if preseason else batter_feature_columns()
        bat_X = extract_features(bat_rows, bat_feature_cols)

        bat_target_preds: dict[str, list[float]] = {}
        for target_name, model in bat_models.items():
            bat_target_preds[target_name] = list(model.predict(bat_X))

        for i, row in enumerate(bat_rows):
            pred: dict[str, Any] = {
                "player_id": row["player_id"],
                "season": row["season"],
                "player_type": "batter",
            }
            for target_name in bat_models:
                pred[target_name] = bat_target_preds[target_name][i]
            predictions_by_row.append(pred)

        # --- Pitcher predictions ---
        pit_fs = build_pitcher_preseason_set(config.seasons) if preseason else build_pitcher_feature_set(config.seasons)
        pit_handle = self._assembler.get_or_materialize(pit_fs)
        pit_rows = self._assembler.read(pit_handle)
        pit_models = load_models(artifact_path / "pitcher_models.joblib")
        pit_feature_cols = pitcher_preseason_feature_columns() if preseason else pitcher_feature_columns()
        pit_X = extract_features(pit_rows, pit_feature_cols)

        pit_target_preds: dict[str, list[float]] = {}
        for target_name, model in pit_models.items():
            pit_target_preds[target_name] = list(model.predict(pit_X))

        for i, row in enumerate(pit_rows):
            pred = {
                "player_id": row["player_id"],
                "season": row["season"],
                "player_type": "pitcher",
            }
            for target_name in pit_models:
                pred[target_name] = pit_target_preds[target_name][i]
            predictions_by_row.append(pred)

        return PredictResult(
            model_name=self.name,
            predictions=predictions_by_row,
            output_path=config.output_dir or config.artifacts_dir,
        )

    def ablate(self, config: ModelConfig) -> AblationResult:
        assert self._assembler is not None, "assembler is required for ablate"
        if len(config.seasons) < 2:
            msg = f"ablate requires at least 2 seasons (got {len(config.seasons)})"
            raise ValueError(msg)

        preseason = self._is_preseason(config)
        train_seasons = config.seasons[:-1]
        holdout_seasons = [config.seasons[-1]]
        feature_impacts: dict[str, float] = {}

        # --- Batter ablation ---
        bat_fs = (
            build_batter_preseason_training_set(config.seasons)
            if preseason
            else build_batter_training_set(config.seasons)
        )
        bat_handle = self._assembler.get_or_materialize(bat_fs)
        bat_splits = self._assembler.split(bat_handle, train=train_seasons, holdout=holdout_seasons)

        bat_train_rows = self._assembler.read(bat_splits.train)
        bat_feature_cols = batter_preseason_feature_columns() if preseason else batter_feature_columns()
        bat_targets = list(BATTER_TARGETS)

        bat_X_train = extract_features(bat_train_rows, bat_feature_cols)
        bat_y_train = extract_targets(bat_train_rows, bat_targets)
        bat_models = fit_models(bat_X_train, bat_y_train, config.model_params)

        if bat_splits.holdout is not None:
            bat_holdout_rows = self._assembler.read(bat_splits.holdout)
            bat_X_holdout = extract_features(bat_holdout_rows, bat_feature_cols)
            bat_y_holdout = extract_targets(bat_holdout_rows, bat_targets)
            bat_importance = compute_permutation_importance(
                bat_models,
                bat_X_holdout,
                bat_y_holdout,
                bat_feature_cols,
            )
            for col, impact in bat_importance.items():
                feature_impacts[f"batter:{col}"] = impact

        # --- Pitcher ablation ---
        pit_fs = (
            build_pitcher_preseason_training_set(config.seasons)
            if preseason
            else build_pitcher_training_set(config.seasons)
        )
        pit_handle = self._assembler.get_or_materialize(pit_fs)
        pit_splits = self._assembler.split(pit_handle, train=train_seasons, holdout=holdout_seasons)

        pit_train_rows = self._assembler.read(pit_splits.train)
        pit_feature_cols = pitcher_preseason_feature_columns() if preseason else pitcher_feature_columns()
        pit_targets = list(PITCHER_TARGETS)

        pit_X_train = extract_features(pit_train_rows, pit_feature_cols)
        pit_y_train = extract_targets(pit_train_rows, pit_targets)
        pit_models = fit_models(pit_X_train, pit_y_train, config.model_params)

        if pit_splits.holdout is not None:
            pit_holdout_rows = self._assembler.read(pit_splits.holdout)
            pit_X_holdout = extract_features(pit_holdout_rows, pit_feature_cols)
            pit_y_holdout = extract_targets(pit_holdout_rows, pit_targets)
            pit_importance = compute_permutation_importance(
                pit_models,
                pit_X_holdout,
                pit_y_holdout,
                pit_feature_cols,
            )
            for col, impact in pit_importance.items():
                feature_impacts[f"pitcher:{col}"] = impact

        return AblationResult(
            model_name=self.name,
            feature_impacts=feature_impacts,
        )

    def _artifact_path(self, config: ModelConfig) -> Path:
        return Path(config.artifacts_dir) / self.name / (config.version or "latest")
