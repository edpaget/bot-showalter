"""Composite engine protocol and implementations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.models.composite.convert import (
    composite_projection_to_domain,
)
from fantasy_baseball_manager.models.composite.targets import BATTER_TARGETS, PITCHER_TARGETS
from fantasy_baseball_manager.models.gbm_training import (
    extract_features,
    extract_targets,
    fit_models,
    score_predictions,
)
from fantasy_baseball_manager.models.marcel.convert import rows_to_marcel_inputs
from fantasy_baseball_manager.models.marcel.engine import age_adjust, regress_to_mean
from fantasy_baseball_manager.models.marcel.types import LeagueAverages, MarcelConfig
from fantasy_baseball_manager.models.statcast_gbm.serialization import save_models


@dataclass(frozen=True)
class EngineConfig:
    marcel_config: MarcelConfig
    projected_season: int
    version: str
    system_name: str


@runtime_checkable
class CompositeEngine(Protocol):
    @property
    def supported_operations(self) -> frozenset[str]: ...

    @property
    def artifact_type(self) -> str: ...

    def predict(
        self,
        bat_rows: list[dict[str, Any]],
        pitch_rows: list[dict[str, Any]],
        bat_pt: dict[int, float],
        pitch_pt: dict[int, float],
        config: EngineConfig,
    ) -> list[dict[str, Any]]: ...


class MarcelEngine:
    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    def predict(
        self,
        bat_rows: list[dict[str, Any]],
        pitch_rows: list[dict[str, Any]],
        bat_pt: dict[int, float],
        pitch_pt: dict[int, float],
        config: EngineConfig,
    ) -> list[dict[str, Any]]:
        marcel_config = config.marcel_config
        bat_lags = len(marcel_config.batting_weights)
        pitch_lags = len(marcel_config.pitching_weights)

        bat_inputs = rows_to_marcel_inputs(bat_rows, marcel_config.batting_categories, bat_lags, pitcher=False)
        pitch_inputs = rows_to_marcel_inputs(pitch_rows, marcel_config.pitching_categories, pitch_lags, pitcher=True)

        predictions: list[dict[str, Any]] = []

        for pid, marcel_input in bat_inputs.items():
            proj_pa = bat_pt.get(pid, 0.0)
            league = LeagueAverages(rates=marcel_input.league_rates)
            regressed = regress_to_mean(
                marcel_input.weighted_rates, league, marcel_input.weighted_pt, marcel_config.batting_regression_pa
            )
            adjusted = age_adjust(regressed, marcel_input.age, marcel_config)
            counting = {cat: adjusted[cat] * proj_pa for cat in adjusted}

            domain = composite_projection_to_domain(
                pid,
                config.projected_season,
                counting,
                adjusted,
                proj_pa,
                pitcher=False,
                version=config.version,
                system=config.system_name,
            )
            predictions.append(
                {
                    "player_id": domain.player_id,
                    "season": domain.season,
                    "player_type": "batter",
                    **domain.stat_json,
                }
            )

        for pid, marcel_input in pitch_inputs.items():
            proj_ip = pitch_pt.get(pid, 0.0)
            league = LeagueAverages(rates=marcel_input.league_rates)
            regressed = regress_to_mean(
                marcel_input.weighted_rates, league, marcel_input.weighted_pt, marcel_config.pitching_regression_ip
            )
            adjusted = age_adjust(regressed, marcel_input.age, marcel_config)
            counting = {cat: adjusted[cat] * proj_ip for cat in adjusted}

            domain = composite_projection_to_domain(
                pid,
                config.projected_season,
                counting,
                adjusted,
                proj_ip,
                pitcher=True,
                version=config.version,
                system=config.system_name,
            )
            predictions.append(
                {
                    "player_id": domain.player_id,
                    "season": domain.season,
                    "player_type": "pitcher",
                    **domain.stat_json,
                }
            )

        return predictions


class GBMEngine:
    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "train", "predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.FILE.value

    def predict(
        self,
        bat_rows: list[dict[str, Any]],
        pitch_rows: list[dict[str, Any]],
        bat_pt: dict[int, float],
        pitch_pt: dict[int, float],
        config: EngineConfig,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def train(
        self,
        bat_train_rows: list[dict[str, Any]],
        bat_holdout_rows: list[dict[str, Any]],
        pitch_train_rows: list[dict[str, Any]],
        pitch_holdout_rows: list[dict[str, Any]],
        bat_feature_cols: list[str],
        pitch_feature_cols: list[str],
        model_params: dict[str, Any],
        artifact_path: Path,
    ) -> dict[str, float]:
        artifact_path.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, float] = {}

        # --- Batter training ---
        batter_params = model_params.get("batter", model_params)
        bat_X = extract_features(bat_train_rows, bat_feature_cols)
        bat_y = extract_targets(bat_train_rows, list(BATTER_TARGETS))
        bat_models = fit_models(bat_X, bat_y, batter_params)

        if bat_holdout_rows:
            bat_X_holdout = extract_features(bat_holdout_rows, bat_feature_cols)
            bat_y_holdout = extract_targets(bat_holdout_rows, list(BATTER_TARGETS))
            bat_metrics = score_predictions(bat_models, bat_X_holdout, bat_y_holdout)
            for key, value in bat_metrics.items():
                metrics[f"batter_{key}"] = value

        save_models(bat_models, artifact_path / "batter_models.joblib")

        # --- Pitcher training ---
        pitcher_params = model_params.get("pitcher", model_params)
        pitch_X = extract_features(pitch_train_rows, pitch_feature_cols)
        pitch_y = extract_targets(pitch_train_rows, list(PITCHER_TARGETS))
        pitch_models = fit_models(pitch_X, pitch_y, pitcher_params)

        if pitch_holdout_rows:
            pitch_X_holdout = extract_features(pitch_holdout_rows, pitch_feature_cols)
            pitch_y_holdout = extract_targets(pitch_holdout_rows, list(PITCHER_TARGETS))
            pitch_metrics = score_predictions(pitch_models, pitch_X_holdout, pitch_y_holdout)
            for key, value in pitch_metrics.items():
                metrics[f"pitcher_{key}"] = value

        save_models(pitch_models, artifact_path / "pitcher_models.joblib")

        return metrics


def resolve_engine(model_params: dict[str, Any]) -> CompositeEngine:
    """Map model_params["engine"] to a CompositeEngine instance."""
    engine_name = model_params.get("engine", "marcel")
    if engine_name == "marcel":
        return MarcelEngine()
    if engine_name == "gbm":
        return GBMEngine()
    msg = f"Unknown engine: {engine_name!r}"
    raise ValueError(msg)
