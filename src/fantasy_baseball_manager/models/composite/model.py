"""Composite model — rate projection using external playing-time model."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fantasy_baseball_manager.features.group_library import (
    make_batting_counting_lags,
    make_batting_rate_lags,
    make_pitching_counting_lags,
)
from fantasy_baseball_manager.features.groups import FeatureGroup, compose_feature_set, get_group
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import FeatureSet
from fantasy_baseball_manager.models.composite.convert import extract_projected_pt
from fantasy_baseball_manager.models.composite.engine import CompositeEngine, EngineConfig, GBMEngine, MarcelEngine
from fantasy_baseball_manager.models.composite.features import (
    append_training_targets,
    batter_target_features,
    feature_columns,
    pitcher_target_features,
)
from fantasy_baseball_manager.models.marcel.features import (
    build_batting_league_averages,
    build_batting_weighted_rates,
    build_pitching_league_averages,
    build_pitching_weighted_rates,
)
from fantasy_baseball_manager.models.marcel.types import MarcelConfig
from fantasy_baseball_manager.models.protocols import (
    ModelConfig,
    PredictResult,
    PrepareResult,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import register

DEFAULT_GROUPS: tuple[str, ...] = (
    "age",
    "projected_batting_pt",
    "projected_pitching_pt",
    "batting_counting_lags",
    "pitching_counting_lags",
)


def _resolve_group(
    name: str,
    marcel_config: MarcelConfig,
    lookup: Callable[[str], FeatureGroup] = get_group,
) -> FeatureGroup:
    """Resolve a feature group by name — static from registry, parameterized from factory."""
    if name == "batting_counting_lags":
        lags = list(range(1, len(marcel_config.batting_weights) + 1))
        return make_batting_counting_lags(marcel_config.batting_categories, lags)
    if name == "pitching_counting_lags":
        lags = list(range(1, len(marcel_config.pitching_weights) + 1))
        return make_pitching_counting_lags(marcel_config.pitching_categories, lags)
    if name == "batting_rate_lags":
        lags = list(range(1, len(marcel_config.batting_weights) + 1))
        return make_batting_rate_lags(("avg", "obp", "slg", "woba"), lags)
    return lookup(name)


def _build_marcel_config(model_params: dict[str, Any]) -> MarcelConfig:
    """Build MarcelConfig from model_params, using defaults for missing keys."""
    kwargs: dict[str, Any] = {}
    for field_name in (
        "batting_weights",
        "pitching_weights",
        "pa_weights",
        "ip_weights",
        "batting_categories",
        "pitching_categories",
    ):
        if field_name in model_params:
            kwargs[field_name] = tuple(model_params[field_name])
    for field_name in (
        "batting_regression_pa",
        "pitching_regression_ip",
        "age_improvement_rate",
        "age_decline_rate",
    ):
        if field_name in model_params:
            kwargs[field_name] = float(model_params[field_name])
    if "age_peak" in model_params:
        kwargs["age_peak"] = int(model_params["age_peak"])
    return MarcelConfig(**kwargs)


@register("composite")
class CompositeModel:
    def __init__(
        self,
        assembler: DatasetAssembler,
        model_name: str = "composite",
        group_lookup: Callable[[str], FeatureGroup] = get_group,
        engine: CompositeEngine | None = None,
    ) -> None:
        self._assembler = assembler
        self._model_name = model_name
        self._get_group = group_lookup
        self._engine: CompositeEngine = engine or MarcelEngine()

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def description(self) -> str:
        return "Rate projection using external playing-time model. Requires playing_time predictions first."

    @property
    def supported_operations(self) -> frozenset[str]:
        return self._engine.supported_operations

    @property
    def artifact_type(self) -> str:
        return self._engine.artifact_type

    def _build_feature_sets(
        self,
        marcel_config: MarcelConfig,
        config: ModelConfig,
    ) -> tuple[FeatureSet, FeatureSet]:
        group_names: tuple[str, ...] = tuple(config.model_params.get("feature_groups", DEFAULT_GROUPS))
        groups = [_resolve_group(name, marcel_config, self._get_group) for name in group_names]

        batter_groups = [g for g in groups if g.player_type in ("batter", "both")]
        pitcher_groups = [g for g in groups if g.player_type in ("pitcher", "both")]

        has_batting_lags = any(g.name == "batting_counting_lags" for g in groups)
        has_pitching_lags = any(g.name == "pitching_counting_lags" for g in groups)

        if has_batting_lags:
            batter_groups.append(
                FeatureGroup(
                    name="batting_transforms",
                    description="Marcel batting transforms (weighted rates + league averages)",
                    player_type="batter",
                    features=(
                        build_batting_weighted_rates(marcel_config.batting_categories, marcel_config.batting_weights),
                        build_batting_league_averages(marcel_config.batting_categories),
                    ),
                )
            )

        if has_pitching_lags:
            pitcher_groups.append(
                FeatureGroup(
                    name="pitching_transforms",
                    description="Marcel pitching transforms (weighted rates + league averages)",
                    player_type="pitcher",
                    features=(
                        build_pitching_weighted_rates(
                            marcel_config.pitching_categories, marcel_config.pitching_weights
                        ),
                        build_pitching_league_averages(marcel_config.pitching_categories),
                    ),
                )
            )

        prefix = self._model_name.replace("-", "_")
        seasons = tuple(config.seasons)
        batting_fs = compose_feature_set(
            name=f"{prefix}_batting",
            groups=batter_groups,
            seasons=seasons,
            source_filter="fangraphs",
        )
        pitching_fs = compose_feature_set(
            name=f"{prefix}_pitching",
            groups=pitcher_groups,
            seasons=seasons,
            source_filter="fangraphs",
        )
        return batting_fs, pitching_fs

    def prepare(self, config: ModelConfig) -> PrepareResult:

        marcel_config = _build_marcel_config(config.model_params)
        batting_fs, pitching_fs = self._build_feature_sets(marcel_config, config)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        return PrepareResult(
            model_name=self.name,
            rows_processed=bat_handle.row_count + pitch_handle.row_count,
            artifacts_path=config.artifacts_dir,
        )

    def train(self, config: ModelConfig) -> TrainResult:
        if len(config.seasons) < 2:
            msg = f"train requires at least 2 seasons (got {len(config.seasons)})"
            raise ValueError(msg)
        if not isinstance(self._engine, GBMEngine):
            msg = f"Engine {type(self._engine).__name__} does not support train"
            raise ValueError(msg)

        marcel_config = _build_marcel_config(config.model_params)
        batting_fs, pitching_fs = self._build_feature_sets(marcel_config, config)

        bat_cols = feature_columns(batting_fs)
        pitch_cols = feature_columns(pitching_fs)

        bat_train_fs = append_training_targets(batting_fs, batter_target_features())
        pitch_train_fs = append_training_targets(pitching_fs, pitcher_target_features())

        bat_handle = self._assembler.get_or_materialize(bat_train_fs)
        pitch_handle = self._assembler.get_or_materialize(pitch_train_fs)

        train_seasons = config.seasons[:-1]
        holdout_seasons = [config.seasons[-1]]

        bat_splits = self._assembler.split(bat_handle, train=train_seasons, holdout=holdout_seasons)
        pitch_splits = self._assembler.split(pitch_handle, train=train_seasons, holdout=holdout_seasons)

        bat_train_rows = self._assembler.read(bat_splits.train)
        bat_holdout_rows = self._assembler.read(bat_splits.holdout) if bat_splits.holdout else []
        pitch_train_rows = self._assembler.read(pitch_splits.train)
        pitch_holdout_rows = self._assembler.read(pitch_splits.holdout) if pitch_splits.holdout else []

        artifact_path = Path(config.artifacts_dir) / self._model_name / (config.version or "latest")

        metrics = self._engine.train(
            bat_train_rows,
            bat_holdout_rows,
            pitch_train_rows,
            pitch_holdout_rows,
            bat_cols,
            pitch_cols,
            config.model_params,
            artifact_path,
        )

        return TrainResult(
            model_name=self.name,
            metrics=metrics,
            artifacts_path=str(artifact_path),
        )

    def predict(self, config: ModelConfig) -> PredictResult:

        marcel_config = _build_marcel_config(config.model_params)
        batting_fs, pitching_fs = self._build_feature_sets(marcel_config, config)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        bat_rows = self._assembler.read(bat_handle)
        pitch_rows = self._assembler.read(pitch_handle)

        bat_pt = extract_projected_pt(bat_rows, pitcher=False)
        pitch_pt = extract_projected_pt(pitch_rows, pitcher=True)

        projected_season = max(config.seasons) + 1 if config.seasons else 2025
        version = config.version or "latest"

        gbm_kwargs: dict[str, Any] = {}
        if isinstance(self._engine, GBMEngine):
            artifact_path = Path(config.artifacts_dir) / self._model_name / (config.version or "latest")
            gbm_kwargs["artifact_path"] = artifact_path
            gbm_kwargs["bat_feature_cols"] = tuple(feature_columns(batting_fs))
            gbm_kwargs["pitch_feature_cols"] = tuple(feature_columns(pitching_fs))

        engine_config = EngineConfig(
            marcel_config=marcel_config,
            projected_season=projected_season,
            version=version,
            system_name=self._model_name,
            **gbm_kwargs,
        )

        predictions = self._engine.predict(bat_rows, pitch_rows, bat_pt, pitch_pt, engine_config)

        return PredictResult(
            model_name=self.name,
            predictions=predictions,
            output_path=config.output_dir or config.artifacts_dir,
        )
