"""Playing-time model — projects PA (batters) and IP (pitchers) via OLS regression."""

from pathlib import Path
from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import AnyFeature, FeatureSet, SpineFilter
from fantasy_baseball_manager.models.playing_time.convert import pt_projection_to_domain
from fantasy_baseball_manager.models.playing_time.engine import (
    fit_playing_time,
    predict_playing_time,
)
from fantasy_baseball_manager.models.playing_time.features import (
    batting_pt_feature_columns,
    build_batting_pt_derived_transforms,
    build_batting_pt_features,
    build_batting_pt_training_features,
    build_pitching_pt_derived_transforms,
    build_pitching_pt_features,
    build_pitching_pt_training_features,
    pitching_pt_feature_columns,
)
from fantasy_baseball_manager.models.playing_time.serialization import (
    load_coefficients,
    save_coefficients,
)
from fantasy_baseball_manager.models.protocols import (
    ModelConfig,
    PredictResult,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import register

_ARTIFACT_FILENAME = "pt_coefficients.joblib"


@register("playing_time")
class PlayingTimeModel:
    def __init__(self, assembler: DatasetAssembler | None = None) -> None:
        self._assembler = assembler

    @property
    def name(self) -> str:
        return "playing_time"

    @property
    def description(self) -> str:
        return "Playing-time projection — projects PA (batters) and IP (pitchers) via OLS regression."

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.FILE.value

    def _build_feature_sets(
        self,
        seasons: list[int],
        *,
        training: bool = False,
        lags: int = 3,
    ) -> tuple[FeatureSet, FeatureSet]:
        if training:
            bat_features = build_batting_pt_training_features(lags)
            pitch_features = build_pitching_pt_training_features(lags)
        else:
            bat_features: list[AnyFeature] = list(build_batting_pt_features(lags))
            bat_features.extend(build_batting_pt_derived_transforms(lags))
            pitch_features: list[AnyFeature] = list(build_pitching_pt_features(lags))
            pitch_features.extend(build_pitching_pt_derived_transforms(lags))

        batting_fs = FeatureSet(
            name="playing_time_batting_train" if training else "playing_time_batting",
            features=tuple(bat_features),
            seasons=tuple(seasons),
            source_filter="fangraphs",
            spine_filter=SpineFilter(min_pa=50, player_type="batter"),
        )
        pitching_fs = FeatureSet(
            name="playing_time_pitching_train" if training else "playing_time_pitching",
            features=tuple(pitch_features),
            seasons=tuple(seasons),
            source_filter="fangraphs",
            spine_filter=SpineFilter(min_ip=10.0, player_type="pitcher"),
        )
        return batting_fs, pitching_fs

    def _artifact_path(self, config: ModelConfig) -> Path:
        return Path(config.artifacts_dir) / self.name / (config.version or "latest")

    def prepare(self, config: ModelConfig) -> PrepareResult:
        assert self._assembler is not None, "assembler is required for prepare"
        lags = config.model_params.get("lags", 3)
        batting_fs, pitching_fs = self._build_feature_sets(config.seasons, lags=lags)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        return PrepareResult(
            model_name=self.name,
            rows_processed=bat_handle.row_count + pitch_handle.row_count,
            artifacts_path=config.artifacts_dir,
        )

    def train(self, config: ModelConfig) -> TrainResult:
        assert self._assembler is not None, "assembler is required for train"
        lags = config.model_params.get("lags", 3)
        batting_fs, pitching_fs = self._build_feature_sets(config.seasons, training=True, lags=lags)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        bat_rows = self._assembler.read(bat_handle)
        pitch_rows = self._assembler.read(pitch_handle)

        bat_columns = batting_pt_feature_columns(lags)
        pitch_columns = pitching_pt_feature_columns(lags)

        bat_coeff = fit_playing_time(bat_rows, bat_columns, "target_pa", "batter")
        pitch_coeff = fit_playing_time(pitch_rows, pitch_columns, "target_ip", "pitcher")

        artifact_path = self._artifact_path(config)
        artifact_path.mkdir(parents=True, exist_ok=True)
        save_coefficients(
            {"batter": bat_coeff, "pitcher": pitch_coeff},
            artifact_path / _ARTIFACT_FILENAME,
        )

        return TrainResult(
            model_name=self.name,
            metrics={
                "r_squared_batter": bat_coeff.r_squared,
                "r_squared_pitcher": pitch_coeff.r_squared,
            },
            artifacts_path=str(artifact_path),
        )

    def predict(self, config: ModelConfig) -> PredictResult:
        assert self._assembler is not None, "assembler is required for predict"
        lags = config.model_params.get("lags", 3)

        artifact_path = self._artifact_path(config)
        coefficients = load_coefficients(artifact_path / _ARTIFACT_FILENAME)
        bat_coeff = coefficients["batter"]
        pitch_coeff = coefficients["pitcher"]

        batting_fs, pitching_fs = self._build_feature_sets(config.seasons, lags=lags)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        bat_rows = self._assembler.read(bat_handle)
        pitch_rows = self._assembler.read(pitch_handle)

        projected_season = max(config.seasons) + 1 if config.seasons else 2025
        version = config.version or "latest"

        predictions: list[dict[str, Any]] = []

        for row in bat_rows:
            pt = predict_playing_time(row, bat_coeff, clamp_min=0.0, clamp_max=750.0)
            domain = pt_projection_to_domain(
                row["player_id"],
                projected_season,
                pt,
                pitcher=False,
                version=version,
            )
            predictions.append(
                {
                    "player_id": domain.player_id,
                    "season": domain.season,
                    "player_type": "batter",
                    **domain.stat_json,
                }
            )

        for row in pitch_rows:
            pt = predict_playing_time(row, pitch_coeff, clamp_min=0.0, clamp_max=250.0)
            domain = pt_projection_to_domain(
                row["player_id"],
                projected_season,
                pt,
                pitcher=True,
                version=version,
            )
            predictions.append(
                {
                    "player_id": domain.player_id,
                    "season": domain.season,
                    "player_type": "pitcher",
                    **domain.stat_json,
                }
            )

        return PredictResult(
            model_name=self.name,
            predictions=predictions,
            output_path=config.output_dir or config.artifacts_dir,
        )
