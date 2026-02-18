"""Composite model — rate projection using external playing-time model."""

from collections.abc import Callable
from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features.group_library import (
    make_batting_counting_lags,
    make_batting_rate_lags,
    make_pitching_counting_lags,
)
from fantasy_baseball_manager.features.groups import FeatureGroup, compose_feature_set, get_group
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import FeatureSet
from fantasy_baseball_manager.models.composite.convert import (
    composite_projection_to_domain,
    extract_projected_pt,
)
from fantasy_baseball_manager.models.marcel.features import (
    build_batting_league_averages,
    build_batting_weighted_rates,
    build_pitching_league_averages,
    build_pitching_weighted_rates,
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
    ) -> None:
        self._assembler = assembler
        self._model_name = model_name
        self._get_group = group_lookup

    @property
    def name(self) -> str:
        return self._model_name

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

    def predict(self, config: ModelConfig) -> PredictResult:

        marcel_config = _build_marcel_config(config.model_params)
        batting_fs, pitching_fs = self._build_feature_sets(marcel_config, config)

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
                pid,
                projected_season,
                counting,
                adjusted,
                proj_pa,
                pitcher=False,
                version=version,
                system=self._model_name,
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
                projected_season,
                counting,
                adjusted,
                proj_ip,
                pitcher=True,
                version=version,
                system=self._model_name,
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
