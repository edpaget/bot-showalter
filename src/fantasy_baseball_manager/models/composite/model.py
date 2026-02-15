"""Composite model — rate projection using external playing-time model."""

from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import FeatureSet
from fantasy_baseball_manager.models.composite.convert import (
    composite_projection_to_domain,
    extract_projected_pt,
)
from fantasy_baseball_manager.models.composite.features import (
    build_composite_batting_features,
    build_composite_pitching_features,
)
from fantasy_baseball_manager.models.marcel.convert import rows_to_marcel_inputs
from fantasy_baseball_manager.models.marcel.engine import age_adjust, regress_to_mean
from fantasy_baseball_manager.models.marcel.types import LeagueAverages, MarcelConfig
from fantasy_baseball_manager.models.protocols import (
    ModelConfig,
    PredictResult,
    PrepareResult,
)
from fantasy_baseball_manager.models.registry import register


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
    def __init__(self, assembler: DatasetAssembler | None = None) -> None:
        self._assembler = assembler

    @property
    def name(self) -> str:
        return "composite"

    @property
    def description(self) -> str:
        return "Rate projection using external playing-time model. Requires playing_time predictions first."

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    def _build_feature_sets(
        self,
        marcel_config: MarcelConfig,
        seasons: list[int],
    ) -> tuple[FeatureSet, FeatureSet]:
        batting_features = build_composite_batting_features(
            marcel_config.batting_categories, marcel_config.batting_weights
        )
        pitching_features = build_composite_pitching_features(
            marcel_config.pitching_categories, marcel_config.pitching_weights
        )

        batting_fs = FeatureSet(
            name="composite_batting",
            features=tuple(batting_features),
            seasons=tuple(seasons),
            source_filter="fangraphs",
        )
        pitching_fs = FeatureSet(
            name="composite_pitching",
            features=tuple(pitching_features),
            seasons=tuple(seasons),
            source_filter="fangraphs",
        )
        return batting_fs, pitching_fs

    def prepare(self, config: ModelConfig) -> PrepareResult:
        assert self._assembler is not None, "assembler is required for prepare"
        marcel_config = _build_marcel_config(config.model_params)
        batting_fs, pitching_fs = self._build_feature_sets(marcel_config, config.seasons)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        return PrepareResult(
            model_name=self.name,
            rows_processed=bat_handle.row_count + pitch_handle.row_count,
            artifacts_path=config.artifacts_dir,
        )

    def predict(self, config: ModelConfig) -> PredictResult:
        assert self._assembler is not None, "assembler is required for predict"
        marcel_config = _build_marcel_config(config.model_params)
        batting_fs, pitching_fs = self._build_feature_sets(marcel_config, config.seasons)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        bat_rows = self._assembler.read(bat_handle)
        pitch_rows = self._assembler.read(pitch_handle)

        projected_season = max(config.seasons) + 1 if config.seasons else 2025
        version = config.version or "latest"

        bat_lags = len(marcel_config.batting_weights)
        pitch_lags = len(marcel_config.pitching_weights)

        bat_inputs = rows_to_marcel_inputs(bat_rows, marcel_config.batting_categories, bat_lags, pitcher=False)
        pitch_inputs = rows_to_marcel_inputs(pitch_rows, marcel_config.pitching_categories, pitch_lags, pitcher=True)

        bat_pt = extract_projected_pt(bat_rows, pitcher=False)
        pitch_pt = extract_projected_pt(pitch_rows, pitcher=True)

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
                pid, projected_season, counting, adjusted, proj_pa, pitcher=False, version=version
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
                pid, projected_season, counting, adjusted, proj_ip, pitcher=True, version=version
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
