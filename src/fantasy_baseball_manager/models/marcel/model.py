from collections.abc import Sequence
from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import Feature, FeatureSet
from fantasy_baseball_manager.models.marcel.convert import (
    projection_to_domain,
    rows_to_player_seasons,
)
from fantasy_baseball_manager.models.marcel.engine import (
    compute_league_averages,
    project_all,
)
from fantasy_baseball_manager.models.marcel.features import (
    build_batting_features,
    build_pitching_features,
)
from fantasy_baseball_manager.models.marcel.types import MarcelConfig
from fantasy_baseball_manager.models.protocols import (
    EvalResult,
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
    batting_features = build_batting_features(marcel_config.batting_categories)
    pitching_features = build_pitching_features(marcel_config.pitching_categories)

    batting_fs = FeatureSet(
        name=f"{name_prefix}_batting",
        features=tuple(batting_features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
    )
    pitching_fs = FeatureSet(
        name=f"{name_prefix}_pitching",
        features=tuple(pitching_features),
        seasons=tuple(seasons),
        source_filter="fangraphs",
    )
    return batting_fs, pitching_fs


@register("marcel")
class MarcelModel:
    def __init__(self, assembler: DatasetAssembler | None = None) -> None:
        self._assembler = assembler

    @property
    def name(self) -> str:
        return "marcel"

    @property
    def description(self) -> str:
        return "Marcel projection system — weighted averages, " "regression to the mean, and aging curves."

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "predict", "evaluate"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    @property
    def declared_features(self) -> tuple[Feature, ...]:
        default_config = MarcelConfig()
        batting = build_batting_features(default_config.batting_categories)
        pitching = build_pitching_features(default_config.pitching_categories)
        return tuple(batting + pitching)

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
        lags = 3

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        bat_rows = self._assembler.read(bat_handle)
        pitch_rows = self._assembler.read(pitch_handle)

        projected_season = max(config.seasons) + 1 if config.seasons else 2025

        batting_cats = list(marcel_config.batting_categories)
        pitching_cats = list(marcel_config.pitching_categories)

        bat_players = rows_to_player_seasons(bat_rows, batting_cats, lags, pitcher=False)
        pitch_players = rows_to_player_seasons(pitch_rows, pitching_cats, lags, pitcher=True)

        # Build player dicts for engine: {player_id: (seasons, age)}
        bat_input: dict[int, tuple[list, int]] = {pid: (seasons, age) for pid, (_, seasons, age) in bat_players.items()}
        pitch_input: dict[int, tuple[list, int]] = {
            pid: (seasons, age) for pid, (_, seasons, age) in pitch_players.items()
        }

        # Compute league averages
        bat_season_map = {pid: seasons for pid, (seasons, _) in bat_input.items()}
        pitch_season_map = {pid: seasons for pid, (seasons, _) in pitch_input.items()}
        bat_league = compute_league_averages(bat_season_map, batting_cats)
        pitch_league = compute_league_averages(pitch_season_map, pitching_cats)

        bat_projections = project_all(bat_input, projected_season, bat_league, marcel_config)
        pitch_projections = project_all(pitch_input, projected_season, pitch_league, marcel_config)

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

    def evaluate(self, config: ModelConfig) -> EvalResult:
        return EvalResult(model_name=self.name, metrics={})
