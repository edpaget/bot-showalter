from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import ArtifactType, EligibilityProvider, GamesLostEstimator, discount_projections
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.models.zar.model import ZarModel
from fantasy_baseball_manager.repos import (  # noqa: TC001 — used in __init__ signature evaluated by inspect.signature()
    PlayerRepo,
    PositionAppearanceRepo,
    ProjectionRepo,
    ValuationRepo,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import LeagueSettings, Projection


def _floor_playing_time(projections: list[Projection], min_pa: int, min_ip: int) -> list[Projection]:
    """Floor discounted PA/IP at eligibility thresholds.

    Prevents borderline players from being excluded by ZarModel's min_pa/min_ip
    filter after injury discounting reduces their playing time below the threshold.
    """
    result: list[Projection] = []
    for proj in projections:
        stats = proj.stat_json
        if proj.player_type == "batter" and stats.get("pa", 0) < min_pa:
            stats = {**stats, "pa": min_pa}
            proj = dataclasses.replace(proj, stat_json=stats)
        elif proj.player_type == "pitcher" and stats.get("ip", 0) < min_ip:
            stats = {**stats, "ip": min_ip}
            proj = dataclasses.replace(proj, stat_json=stats)
        result.append(proj)
    return result


@register("zar-injury-risk")
class ZarInjuryRiskModel:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        position_repo: PositionAppearanceRepo,
        player_repo: PlayerRepo | None = None,
        valuation_repo: ValuationRepo | None = None,
        eligibility_service: EligibilityProvider | None = None,
        injury_profiler: GamesLostEstimator | None = None,
    ) -> None:
        self._zar = ZarModel(
            projection_repo,
            position_repo,
            player_repo,
            valuation_repo,
            eligibility_service,
        )
        self._profiler = injury_profiler
        self._projection_repo = projection_repo

    @property
    def name(self) -> str:
        return "zar-injury-risk"

    @property
    def description(self) -> str:
        return "Injury-risk-adjusted ZAR valuations"

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

        season = config.seasons[0] if config.seasons else 2025
        seasons_back: int = config.model_params.get("seasons_back", 5)
        proj_system: str = config.model_params["projection_system"]
        proj_version: str | None = config.model_params.get("projection_version")

        # 1. Build injury map
        season_list = list(range(season - seasons_back, season))
        estimates = self._profiler.list_games_lost_estimates(season_list, projection_season=season)
        injury_map: dict[int, float] = {est.player_id: est.expected_days_lost for est, _, _ in estimates}

        # 2. Read and discount projections
        projections: list[Projection] = config.model_params.get("projections") or []
        if not projections:
            if proj_version is not None:
                projections = self._projection_repo.get_by_system_version(proj_system, proj_version)
                projections = [p for p in projections if p.season == season]
            else:
                projections = self._projection_repo.get_by_season(season, system=proj_system)
        discounted = discount_projections(projections, injury_map)

        # Floor discounted PA/IP at eligibility thresholds so borderline players
        # aren't dropped from the pool by ZarModel's min_pa/min_ip filter.
        league: LeagueSettings = config.model_params["league"]
        min_pa = league.eligibility.min_pa or 1
        min_ip = league.eligibility.min_ip or 1
        discounted = _floor_playing_time(discounted, min_pa, min_ip)

        # 3. Build config for ZarModel with injury-risk system name and discounted projections
        zar_params = dict(config.model_params)
        zar_params["valuation_system"] = "zar-injury-risk"
        zar_params["projections"] = discounted
        zar_config = dataclasses.replace(config, model_params=zar_params)

        # 4. Delegate to ZarModel
        result = self._zar.predict(zar_config)

        # 5. Return result with our model name
        return PredictResult(
            model_name=self.name,
            predictions=result.predictions,
            output_path=result.output_path,
        )
