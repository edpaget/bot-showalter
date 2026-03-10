from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import ArtifactType, EligibilityProvider
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
    from fantasy_baseball_manager.domain import LeagueSettings


REFORMED_WEIGHTS: dict[str, float] = {"sv+hld": 0.0}
REFORMED_MIN_IP = 60


@register("zar-reformed")
class ZarReformedModel:
    """ZAR with SV+HLD removed and min_ip raised to 60."""

    def __init__(
        self,
        projection_repo: ProjectionRepo,
        position_repo: PositionAppearanceRepo,
        player_repo: PlayerRepo | None = None,
        valuation_repo: ValuationRepo | None = None,
        eligibility_service: EligibilityProvider | None = None,
    ) -> None:
        self._zar = ZarModel(
            projection_repo,
            position_repo,
            player_repo,
            valuation_repo,
            eligibility_service,
        )

    @property
    def name(self) -> str:
        return "zar-reformed"

    @property
    def description(self) -> str:
        return "ZAR with SV+HLD removed and min_ip raised to 60"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    def predict(self, config: ModelConfig) -> PredictResult:
        league: LeagueSettings = config.model_params["league"]

        # Override min_ip to at least REFORMED_MIN_IP
        current_min_ip = league.eligibility.min_ip or 0
        if current_min_ip < REFORMED_MIN_IP:
            new_eligibility = dataclasses.replace(league.eligibility, min_ip=REFORMED_MIN_IP)
            league = dataclasses.replace(league, eligibility=new_eligibility)

        # Build modified config with category weights and reformed league
        zar_params = dict(config.model_params)
        zar_params["valuation_system"] = "zar-reformed"
        zar_params["category_weights"] = REFORMED_WEIGHTS
        zar_params["league"] = league
        zar_config = dataclasses.replace(config, model_params=zar_params)

        result = self._zar.predict(zar_config)

        return PredictResult(
            model_name=self.name,
            predictions=result.predictions,
            output_path=result.output_path,
        )
