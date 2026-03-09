from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    ArtifactType,
    EligibilityProvider,
    GamesLostEstimator,
    Position,
    ReplacementPadder,
)
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.models.zar.model import ZarModel, _extract_stats
from fantasy_baseball_manager.models.zar.positions import build_roster_spots
from fantasy_baseball_manager.repos import (  # noqa: TC001 — used in __init__ signature evaluated by inspect.signature()
    PlayerRepo,
    PositionAppearanceRepo,
    ProjectionRepo,
    ValuationRepo,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import LeagueSettings, Projection, ReplacementProfile


@register("zar-replacement-padded")
class ZarReplacementPaddedModel:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        position_repo: PositionAppearanceRepo,
        player_repo: PlayerRepo | None = None,
        valuation_repo: ValuationRepo | None = None,
        eligibility_service: EligibilityProvider | None = None,
        injury_profiler: GamesLostEstimator | None = None,
        replacement_padder: ReplacementPadder | None = None,
    ) -> None:
        self._zar = ZarModel(
            projection_repo,
            position_repo,
            player_repo,
            valuation_repo,
            eligibility_service,
        )
        self._profiler = injury_profiler
        self._padder = replacement_padder
        self._projection_repo = projection_repo
        self._eligibility_service = eligibility_service

    @property
    def name(self) -> str:
        return "zar-replacement-padded"

    @property
    def description(self) -> str:
        return "Replacement-padded ZAR valuations"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    def predict(self, config: ModelConfig) -> PredictResult:
        if self._profiler is None:
            msg = "injury_profiler is required for predict()"
            raise TypeError(msg)
        if self._eligibility_service is None:
            msg = "eligibility_service is required for predict()"
            raise TypeError(msg)
        if self._padder is None:
            msg = "replacement_padder is required for predict()"
            raise TypeError(msg)

        season = config.seasons[0] if config.seasons else 2025
        seasons_back: int = config.model_params.get("seasons_back", 5)
        proj_system: str = config.model_params["projection_system"]
        proj_version: str | None = config.model_params.get("projection_version")
        league: LeagueSettings = config.model_params["league"]

        # 1. Build injury map
        season_list = list(range(season - seasons_back, season))
        estimates = self._profiler.list_games_lost_estimates(season_list, projection_season=season)
        injury_map: dict[int, float] = {est.player_id: est.expected_days_lost for est, _, _ in estimates}

        # 2. Read projections
        projections: list[Projection] = config.model_params.get("projections") or []
        if not projections:
            if proj_version is not None:
                projections = self._projection_repo.get_by_system_version(proj_system, proj_version)
                projections = [p for p in projections if p.season == season]
            else:
                projections = self._projection_repo.get_by_season(season, system=proj_system)

        # 3. Get position maps
        batter_position_map = self._eligibility_service.get_batter_positions(season, league)
        min_pa = league.eligibility.min_pa or 1
        min_ip = league.eligibility.min_ip or 1

        # 4. Split into batters and pitchers
        batter_projs = [p for p in projections if p.player_type == "batter" and p.stat_json.get("pa", 0) >= min_pa]
        pitcher_projs = [p for p in projections if p.player_type == "pitcher" and p.stat_json.get("ip", 0) >= min_ip]

        pitcher_position_map = self._eligibility_service.get_pitcher_positions(
            season, league, [p.player_id for p in pitcher_projs]
        )

        # 5. Pass 1: Compute replacement profiles
        batter_roster_spots = build_roster_spots(league)
        if league.roster_util > 0:
            no_pos_batter: list[str] = ["UTIL"]
        else:
            no_pos_batter = []
        batter_player_positions = [batter_position_map.get(p.player_id, no_pos_batter) for p in batter_projs]
        batter_stats = _extract_stats(batter_projs)
        batter_profiles = self._padder.compute_replacement_profiles(
            batter_stats,
            batter_player_positions,
            batter_roster_spots,
            league.teams,
            list(league.batting_categories),
            "batter",
        )

        if league.pitcher_positions:
            pitcher_roster_spots = dict(league.pitcher_positions)
        else:
            pitcher_roster_spots = {Position.P: league.roster_pitchers}
        pitcher_player_positions = [pitcher_position_map.get(p.player_id, ["P"]) for p in pitcher_projs]
        pitcher_stats = _extract_stats(pitcher_projs)
        pitcher_profiles = self._padder.compute_replacement_profiles(
            pitcher_stats,
            pitcher_player_positions,
            pitcher_roster_spots,
            league.teams,
            list(league.pitching_categories),
            "pitcher",
        )

        # 6. Pass 2: Blend projections
        all_profiles: dict[str, ReplacementProfile] = {**batter_profiles, **pitcher_profiles}
        all_position_map: dict[int, list[str]] = {**batter_position_map, **pitcher_position_map}
        blended = self._padder.blend_projections(projections, all_profiles, injury_map, all_position_map)

        # 7. Pass 3: Final ZAR on blended projections
        zar_params = dict(config.model_params)
        zar_params["valuation_system"] = "zar-replacement-padded"
        zar_params["projections"] = blended
        zar_config = dataclasses.replace(config, model_params=zar_params)
        result = self._zar.predict(zar_config)

        return PredictResult(
            model_name=self.name,
            predictions=result.predictions,
            output_path=result.output_path,
        )
