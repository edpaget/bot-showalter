"""Playing-time model — projects PA (batters) and IP (pitchers) only."""

from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.features.protocols import DatasetAssembler
from fantasy_baseball_manager.features.types import FeatureSet
from fantasy_baseball_manager.models.marcel.convert import rows_to_player_seasons
from fantasy_baseball_manager.models.marcel.engine import project_playing_time
from fantasy_baseball_manager.models.marcel.types import MarcelConfig
from fantasy_baseball_manager.models.playing_time.convert import pt_projection_to_domain
from fantasy_baseball_manager.models.playing_time.features import (
    build_batting_pt_features,
    build_pitching_pt_features,
)
from fantasy_baseball_manager.models.protocols import (
    ModelConfig,
    PredictResult,
    PrepareResult,
)
from fantasy_baseball_manager.models.registry import register


def _build_marcel_config(model_params: dict[str, Any]) -> MarcelConfig:
    """Build MarcelConfig from model_params, using defaults for missing keys."""
    kwargs: dict[str, Any] = {}
    for field_name in ("pa_weights", "ip_weights"):
        if field_name in model_params:
            kwargs[field_name] = tuple(model_params[field_name])
    for field_name in (
        "batting_baseline_pa",
        "pitching_starter_baseline_ip",
        "pitching_reliever_baseline_ip",
        "reliever_gs_ratio",
    ):
        if field_name in model_params:
            kwargs[field_name] = float(model_params[field_name])
    return MarcelConfig(**kwargs)


@register("playing_time")
class PlayingTimeModel:
    def __init__(self, assembler: DatasetAssembler | None = None) -> None:
        self._assembler = assembler

    @property
    def name(self) -> str:
        return "playing_time"

    @property
    def description(self) -> str:
        return "Playing-time projection — projects PA (batters) and IP (pitchers) using weighted historical data."

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
        bat_lags = len(marcel_config.pa_weights)
        pitch_lags = len(marcel_config.ip_weights)

        batting_features = build_batting_pt_features(lags=bat_lags)
        pitching_features = build_pitching_pt_features(lags=pitch_lags)

        batting_fs = FeatureSet(
            name="playing_time_batting",
            features=tuple(batting_features),
            seasons=tuple(seasons),
            source_filter="fangraphs",
        )
        pitching_fs = FeatureSet(
            name="playing_time_pitching",
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
        bat_lags = len(marcel_config.pa_weights)
        pitch_lags = len(marcel_config.ip_weights)

        bat_handle = self._assembler.get_or_materialize(batting_fs)
        pitch_handle = self._assembler.get_or_materialize(pitching_fs)

        bat_rows = self._assembler.read(bat_handle)
        pitch_rows = self._assembler.read(pitch_handle)

        projected_season = max(config.seasons) + 1 if config.seasons else 2025
        version = config.version or "latest"

        bat_players = rows_to_player_seasons(bat_rows, categories=[], lags=bat_lags, pitcher=False)
        pitch_players = rows_to_player_seasons(pitch_rows, categories=[], lags=pitch_lags, pitcher=True)

        predictions: list[dict[str, Any]] = []

        for pid, (_pid, season_lines, _age) in bat_players.items():
            pt = project_playing_time(season_lines, marcel_config)
            domain = pt_projection_to_domain(pid, projected_season, pt, pitcher=False, version=version)
            predictions.append(
                {
                    "player_id": domain.player_id,
                    "season": domain.season,
                    "player_type": "batter",
                    **domain.stat_json,
                }
            )

        for pid, (_pid, season_lines, _age) in pitch_players.items():
            pt = project_playing_time(season_lines, marcel_config)
            domain = pt_projection_to_domain(pid, projected_season, pt, pitcher=True, version=version)
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
