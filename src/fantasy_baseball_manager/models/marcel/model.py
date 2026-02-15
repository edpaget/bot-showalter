from collections.abc import Sequence
from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import AnyFeature, FeatureSet
from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.models.marcel.convert import (
    projection_to_domain,
    rows_to_marcel_inputs,
)
from fantasy_baseball_manager.models.marcel.engine import project_all
from fantasy_baseball_manager.repos.protocols import ProjectionRepo
from fantasy_baseball_manager.models.marcel.features import (
    build_batting_features,
    build_batting_league_averages,
    build_batting_weighted_rates,
    build_pitching_features,
    build_pitching_league_averages,
    build_pitching_weighted_rates,
)
from fantasy_baseball_manager.models.marcel.types import MarcelConfig
from fantasy_baseball_manager.models.protocols import (
    Evaluator,
    ModelConfig,
    PredictResult,
    PrepareResult,
)
from fantasy_baseball_manager.models.registry import register


def _is_batter(row: dict[str, Any]) -> bool:
    pos = row.get("position")
    if pos is None:
        return True
    return any(p != "P" for p in pos.split(","))


def _is_pitcher(row: dict[str, Any]) -> bool:
    pos = row.get("position")
    if pos is None:
        return True
    return "P" in pos.split(",")


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
        "batting_baseline_pa",
        "pitching_starter_baseline_ip",
        "pitching_reliever_baseline_ip",
        "age_improvement_rate",
        "age_decline_rate",
        "reliever_gs_ratio",
    ):
        if field_name in model_params:
            kwargs[field_name] = float(model_params[field_name])
    if "age_peak" in model_params:
        kwargs["age_peak"] = int(model_params["age_peak"])
    return MarcelConfig(**kwargs)


def _build_feature_sets(
    marcel_config: MarcelConfig,
    seasons: Sequence[int],
    name_prefix: str,
) -> tuple[FeatureSet, FeatureSet]:
    """Build batting and pitching FeatureSets from MarcelConfig."""
    batting_base = build_batting_features(marcel_config.batting_categories)
    batting_weighted = build_batting_weighted_rates(marcel_config.batting_categories, marcel_config.batting_weights)
    batting_league = build_batting_league_averages(marcel_config.batting_categories)

    pitching_base = build_pitching_features(marcel_config.pitching_categories)
    pitching_weighted = build_pitching_weighted_rates(marcel_config.pitching_categories, marcel_config.pitching_weights)
    pitching_league = build_pitching_league_averages(marcel_config.pitching_categories)

    batting_fs = FeatureSet(
        name=f"{name_prefix}_batting",
        features=(*batting_base, batting_weighted, batting_league),
        seasons=tuple(seasons),
        source_filter="fangraphs",
    )
    pitching_fs = FeatureSet(
        name=f"{name_prefix}_pitching",
        features=(*pitching_base, pitching_weighted, pitching_league),
        seasons=tuple(seasons),
        source_filter="fangraphs",
    )
    return batting_fs, pitching_fs


@register("marcel")
class MarcelModel:
    def __init__(
        self,
        assembler: DatasetAssembler | None = None,
        evaluator: Evaluator | None = None,
        projection_repo: ProjectionRepo | None = None,
    ) -> None:
        self._assembler = assembler
        self._evaluator = evaluator
        self._projection_repo = projection_repo

    @property
    def name(self) -> str:
        return "marcel"

    @property
    def description(self) -> str:
        return "Marcel projection system — weighted averages, regression to the mean, and aging curves."

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "predict", "evaluate"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    @property
    def declared_features(self) -> tuple[AnyFeature, ...]:
        default_config = MarcelConfig()
        batting_fs, pitching_fs = _build_feature_sets(default_config, [], "marcel")
        return batting_fs.features + pitching_fs.features

    def prepare(self, config: ModelConfig) -> PrepareResult:
        assert self._assembler is not None, "assembler is required for prepare"
        marcel_config = _build_marcel_config(config.model_params)
        batting_fs, pitching_fs = _build_feature_sets(marcel_config, config.seasons, self.name)

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
        batting_fs, pitching_fs = _build_feature_sets(marcel_config, config.seasons, self.name)
        lags = len(marcel_config.batting_weights)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        bat_rows = [r for r in self._assembler.read(bat_handle) if _is_batter(r)]
        pitch_rows = [r for r in self._assembler.read(pitch_handle) if _is_pitcher(r)]

        projected_season = max(config.seasons) + 1 if config.seasons else 2025

        bat_inputs = rows_to_marcel_inputs(bat_rows, marcel_config.batting_categories, lags, pitcher=False)
        pitch_inputs = rows_to_marcel_inputs(
            pitch_rows, marcel_config.pitching_categories, len(marcel_config.pitching_weights), pitcher=True
        )

        pt_lookup: dict[int, float] | None = None
        if self._projection_repo is not None:
            pt_projections = self._projection_repo.get_by_season(projected_season, system="playing_time")
            pt_map: dict[int, float] = {}
            for proj in pt_projections:
                stat = proj.stat_json
                if proj.player_type == "batter" and "pa" in stat:
                    pt_map[proj.player_id] = float(stat["pa"])
                elif proj.player_type == "pitcher" and "ip" in stat:
                    pt_map[proj.player_id] = float(stat["ip"])
            if pt_map:
                pt_lookup = pt_map

        bat_projections = project_all(bat_inputs, projected_season, marcel_config, projected_pts=pt_lookup)
        pitch_projections = project_all(pitch_inputs, projected_season, marcel_config, projected_pts=pt_lookup)

        version = config.version or "latest"

        predictions: list[dict[str, Any]] = []
        for proj in bat_projections:
            domain = projection_to_domain(proj, version, "batter")
            predictions.append(
                {
                    "player_id": domain.player_id,
                    "season": domain.season,
                    "player_type": "batter",
                    **domain.stat_json,
                }
            )
        for proj in pitch_projections:
            domain = projection_to_domain(proj, version, "pitcher")
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

    def evaluate(self, config: ModelConfig) -> SystemMetrics:
        assert self._evaluator is not None, "evaluator is required for evaluate"
        version = config.version or "latest"
        season = config.seasons[0]
        return self._evaluator.evaluate(self.name, version, season, top=config.top)
