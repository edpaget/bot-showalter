"""MLE projection model — translates minor league stats to MLB-equivalent projections."""

from collections import defaultdict
from typing import Any

from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.models.mle.engine import (
    apply_recency_weights,
    clamp_xbh,
    combine_translated_lines,
    regress_to_mlb,
    translate_batting_line,
)
from fantasy_baseball_manager.models.mle.types import (
    AgeAdjustmentConfig,
    MLEConfig,
    TranslatedBattingLine,
)
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult, PrepareResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.repos.protocols import (
    LeagueEnvironmentRepo,
    LevelFactorRepo,
    MinorLeagueBattingStatsRepo,
)


@register("mle")
class MLEModel:
    def __init__(
        self,
        milb_repo: MinorLeagueBattingStatsRepo | None = None,
        league_env_repo: LeagueEnvironmentRepo | None = None,
        level_factor_repo: LevelFactorRepo | None = None,
    ) -> None:
        self._milb_repo = milb_repo
        self._league_env_repo = league_env_repo
        self._level_factor_repo = level_factor_repo

    @property
    def name(self) -> str:
        return "mle"

    @property
    def description(self) -> str:
        return "Minor league equivalency projection model"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"prepare", "predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    def prepare(self, config: ModelConfig) -> PrepareResult:
        assert self._milb_repo is not None, "milb_repo is required for prepare"
        assert self._level_factor_repo is not None, "level_factor_repo is required for prepare"

        params = config.model_params
        season: int = params.get("season", config.seasons[0] if config.seasons else 0)

        # Discover which levels have configured factors
        level_factors = self._level_factor_repo.get_by_season(season)
        player_ids: set[int] = set()
        for lf in level_factors:
            stats = self._milb_repo.get_by_season_level(season, lf.level)
            for s in stats:
                player_ids.add(s.player_id)

        output_path = config.output_dir or config.artifacts_dir
        return PrepareResult(
            model_name="mle",
            rows_processed=len(player_ids),
            artifacts_path=output_path,
        )

    def predict(self, config: ModelConfig) -> PredictResult:
        assert self._milb_repo is not None, "milb_repo is required for predict"
        assert self._league_env_repo is not None, "league_env_repo is required for predict"
        assert self._level_factor_repo is not None, "level_factor_repo is required for predict"

        params = config.model_params
        projected_season: int = params["season"]
        mle_seasons: list[int] = params.get("mle_seasons", config.seasons)
        mle_config = MLEConfig()
        version: str = params.get("version", "v1")

        # Parse age config from params if provided
        age_config: AgeAdjustmentConfig | None = None
        age_config_dict: dict[str, Any] | None = params.get("age_config")
        if age_config_dict is not None:
            age_config = AgeAdjustmentConfig(**age_config_dict)

        # Collect all translated lines per player per season
        player_season_lines: dict[int, dict[int, list[TranslatedBattingLine]]] = defaultdict(lambda: defaultdict(list))
        player_ages: dict[int, float] = {}

        for season in mle_seasons:
            self._translate_season(season, mle_config, age_config, player_season_lines, player_ages)

        # For each player: combine multi-level, apply recency, regress, output
        predictions: list[dict[str, Any]] = []
        for player_id, season_map in player_season_lines.items():
            projection = self._project_player(
                player_id=player_id,
                season_map=season_map,
                mle_seasons=mle_seasons,
                projected_season=projected_season,
                mle_config=mle_config,
                version=version,
                age=player_ages.get(player_id),
            )
            if projection is not None:
                predictions.append(projection)

        output_path = config.output_dir or config.artifacts_dir
        return PredictResult(
            model_name="mle",
            predictions=predictions,
            output_path=output_path,
        )

    def _translate_season(
        self,
        season: int,
        mle_config: MLEConfig,
        age_config: AgeAdjustmentConfig | None,
        player_season_lines: dict[int, dict[int, list[TranslatedBattingLine]]],
        player_ages: dict[int, float] | None = None,
    ) -> None:
        assert self._milb_repo is not None
        assert self._league_env_repo is not None
        assert self._level_factor_repo is not None

        level_factors = self._level_factor_repo.get_by_season(season)

        # Get MLB environment for this season
        mlb_envs = self._league_env_repo.get_by_season_level(season, "MLB")
        if not mlb_envs:
            return
        mlb_env = mlb_envs[0]

        for lf in level_factors:
            # Get league environment for this level
            level_envs = self._league_env_repo.get_by_season_level(season, lf.level)
            if not level_envs:
                continue
            level_env = level_envs[0]

            stats_list = self._milb_repo.get_by_season_level(season, lf.level)
            for stats in stats_list:
                if stats.pa < mle_config.min_pa:
                    continue
                translated = translate_batting_line(
                    stats=stats,
                    league_env=level_env,
                    mlb_env=mlb_env,
                    level_factor=lf,
                    config=mle_config,
                    age_config=age_config,
                )
                player_season_lines[stats.player_id][season].append(translated)
                if player_ages is not None:
                    player_ages[stats.player_id] = stats.age

    def _project_player(
        self,
        player_id: int,
        season_map: dict[int, list[TranslatedBattingLine]],
        mle_seasons: list[int],
        projected_season: int,
        mle_config: MLEConfig,
        version: str,
        age: float | None = None,
    ) -> dict[str, Any] | None:
        assert self._league_env_repo is not None

        # Combine multi-level lines within each season
        combined_by_season: list[tuple[TranslatedBattingLine, float]] = []
        for i, season in enumerate(sorted(mle_seasons, reverse=True)):
            if season not in season_map:
                continue
            lines = season_map[season]
            combined = combine_translated_lines(lines)
            weight = mle_config.season_weights[i] if i < len(mle_config.season_weights) else 0.0
            combined_by_season.append((combined, weight))

        if not combined_by_season:
            return None

        # Apply recency weights
        weighted_line = apply_recency_weights(combined_by_season)

        # Compute effective PA and regress toward MLB average
        total_mle_pa = sum(line.pa * w for line, w in combined_by_season if w > 0)
        effective_pa = total_mle_pa * mle_config.discount_factor

        # Get MLB environment for regression target (use most recent season)
        most_recent = max(s for s in season_map)
        mlb_envs = self._league_env_repo.get_by_season_level(most_recent, "MLB")
        if not mlb_envs:
            return None
        mlb_env = mlb_envs[0]

        player_rates = {
            "k_pct": weighted_line.k_pct,
            "bb_pct": weighted_line.bb_pct,
            "iso": weighted_line.iso,
            "babip": weighted_line.babip,
        }
        mlb_rates = {
            "k_pct": mlb_env.k_pct,
            "bb_pct": mlb_env.bb_pct,
            "iso": mlb_env.slg - mlb_env.avg,
            "babip": mlb_env.babip,
        }

        regressed = regress_to_mlb(player_rates, mlb_rates, effective_pa, mle_config.regression_pa)

        return self._build_prediction(
            player_id=player_id,
            projected_season=projected_season,
            weighted_line=weighted_line,
            regressed_rates=regressed,
            version=version,
            age=age,
        )

    def _build_prediction(
        self,
        player_id: int,
        projected_season: int,
        weighted_line: TranslatedBattingLine,
        regressed_rates: dict[str, float],
        version: str,
        age: float | None = None,
    ) -> dict[str, Any]:
        # Reconstruct counting stats from regressed rates
        pa = weighted_line.pa
        so = round(regressed_rates["k_pct"] * pa)
        bb = round(regressed_rates["bb_pct"] * pa)
        hbp = weighted_line.hbp
        sf = weighted_line.sf
        ab = pa - bb - hbp - sf
        bip = max(ab - so + sf, 0)
        h_from_bip = round(regressed_rates["babip"] * bip) if bip > 0 else 0
        hr = max(round(regressed_rates["iso"] * ab / 3.0), 0) if ab > 0 else 0
        h = h_from_bip + hr
        h = min(h, ab)

        # Preserve XBH ratios from weighted line
        if weighted_line.h > 0:
            doubles_ratio = weighted_line.doubles / weighted_line.h
            triples_ratio = weighted_line.triples / weighted_line.h
        else:
            doubles_ratio = 0.0
            triples_ratio = 0.0

        doubles = round(h * doubles_ratio)
        triples = round(h * triples_ratio)

        doubles, triples = clamp_xbh(doubles, triples, hr, h)

        line = TranslatedBattingLine(
            player_id=player_id,
            season=projected_season,
            source_level=weighted_line.source_level,
            pa=pa,
            ab=ab,
            h=h,
            doubles=doubles,
            triples=triples,
            hr=hr,
            bb=bb,
            so=so,
            hbp=hbp,
            sf=sf,
        )

        result: dict[str, Any] = {
            "player_id": player_id,
            "season": projected_season,
            "player_type": "batter",
            "system": "mle",
            "version": version,
            "pa": pa,
            "ab": ab,
            "h": h,
            "doubles": doubles,
            "triples": triples,
            "hr": hr,
            "bb": bb,
            "so": so,
            "avg": round(line.avg, 3),
            "obp": round(line.obp, 3),
            "slg": round(line.slg, 3),
            "iso": round(line.iso, 3),
            "babip": round(line.babip, 3),
            "k_pct": round(line.k_pct, 3),
            "bb_pct": round(line.bb_pct, 3),
        }
        if age is not None:
            result["age"] = age
        return result
