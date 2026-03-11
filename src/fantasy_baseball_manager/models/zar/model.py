import dataclasses
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import ArtifactType, EligibilityProvider, Position, Valuation
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.models.zar.engine import (
    compute_budget_split,
    run_zar_pipeline,
)
from fantasy_baseball_manager.models.zar.positions import (
    best_position,
    build_roster_spots,
)
from fantasy_baseball_manager.repos import (  # noqa: TC001 — used in __init__ signature evaluated by inspect.signature()
    PlayerRepo,
    PositionAppearanceRepo,
    ProjectionRepo,
    ValuationRepo,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import LeagueSettings, Projection


def _extract_stats(projections: list[Projection]) -> list[dict[str, float]]:
    """Convert projection stat_json dicts to float-valued dicts."""
    result: list[dict[str, float]] = []
    for proj in projections:
        row: dict[str, float] = {}
        for k, v in proj.stat_json.items():
            if isinstance(v, int | float):
                row[k] = float(v)
        result.append(row)
    return result


@register("zar")
class ZarModel:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        position_repo: PositionAppearanceRepo,
        player_repo: PlayerRepo | None = None,
        valuation_repo: ValuationRepo | None = None,
        eligibility_service: EligibilityProvider | None = None,
    ) -> None:
        self._projection_repo = projection_repo
        self._player_repo = player_repo
        self._position_repo = position_repo
        self._valuation_repo = valuation_repo
        self._eligibility_service = eligibility_service

    @property
    def name(self) -> str:
        return "zar"

    @property
    def description(self) -> str:
        return "Z-Score Above Replacement valuation for H2H categories leagues"

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset({"predict"})

    @property
    def artifact_type(self) -> str:
        return ArtifactType.NONE.value

    def predict(self, config: ModelConfig) -> PredictResult:
        if self._eligibility_service is None:
            msg = "eligibility_service is required for predict()"
            raise TypeError(msg)
        league: LeagueSettings = config.model_params["league"]
        proj_system: str = config.model_params["projection_system"]
        proj_version: str | None = config.model_params.get("projection_version")
        valuation_system: str = config.model_params.get("valuation_system", "zar")
        season = config.seasons[0] if config.seasons else 2025
        version = config.version or "1.0"

        # 1. Read projections (use pre-loaded if provided, otherwise read from repo)
        projections: list[Projection] = config.model_params.get("projections") or []
        if not projections:
            if proj_version is not None:
                projections = self._projection_repo.get_by_system_version(proj_system, proj_version)
                projections = [p for p in projections if p.season == season]
            else:
                projections = self._projection_repo.get_by_season(season, system=proj_system)
        position_map = self._eligibility_service.get_batter_positions(season, league)

        # 2. Split into batters and pitchers (apply playing-time cutoffs)
        min_pa = league.eligibility.min_pa or 1
        min_ip = league.eligibility.min_ip or 1
        batter_projs = [p for p in projections if p.player_type == "batter" and p.stat_json.get("pa", 0) >= min_pa]
        pitcher_projs = [p for p in projections if p.player_type == "pitcher" and p.stat_json.get("ip", 0) >= min_ip]

        # 3. Budget split proportional to category count
        batter_budget, pitcher_budget = compute_budget_split(league)

        # 3.5a. Category weights (optional, for ZAR reform)
        category_weights: dict[str, float] | None = config.model_params.get("category_weights")
        use_direct_rates: bool = config.model_params.get("use_direct_rates", False)
        use_optimal_assignment: bool = config.model_params.get("use_optimal_assignment", True)

        # 3.5. Variance correction: split pre-computed stdev overrides by pool
        batter_stdev_overrides: dict[str, float] | None = None
        pitcher_stdev_overrides: dict[str, float] | None = None
        all_stdevs: dict[str, float] | None = config.model_params.get("_stdev_overrides")
        if all_stdevs:
            batting_keys = {c.key for c in league.batting_categories}
            pitching_keys = {c.key for c in league.pitching_categories}
            batter_stdev_overrides = {k: v for k, v in all_stdevs.items() if k in batting_keys}
            pitcher_stdev_overrides = {k: v for k, v in all_stdevs.items() if k in pitching_keys}

        # 4. Run ZAR for batters
        batter_valuations = self._value_pool(
            batter_projs,
            list(league.batting_categories),
            position_map,
            league,
            batter_budget,
            "batter",
            season,
            version,
            proj_system,
            stdev_overrides=batter_stdev_overrides,
            category_weights=category_weights,
            use_direct_rates=use_direct_rates,
            system=valuation_system,
            use_optimal_assignment=use_optimal_assignment,
        )

        # 5. Run ZAR for pitchers
        pitcher_position_map = self._eligibility_service.get_pitcher_positions(
            season, league, [p.player_id for p in pitcher_projs], projections=pitcher_projs
        )
        if league.pitcher_positions:
            pitcher_roster_spots = dict(league.pitcher_positions)
        else:
            pitcher_roster_spots = {Position.P: league.roster_pitchers}
        pitcher_valuations = self._value_pool(
            pitcher_projs,
            list(league.pitching_categories),
            pitcher_position_map,
            league,
            pitcher_budget,
            "pitcher",
            season,
            version,
            proj_system,
            pitcher_roster_spots=pitcher_roster_spots,
            stdev_overrides=pitcher_stdev_overrides,
            category_weights=category_weights,
            use_direct_rates=use_direct_rates,
            system=valuation_system,
            use_optimal_assignment=use_optimal_assignment,
        )

        # 6. Rank all valuations combined by value descending
        all_valuations = batter_valuations + pitcher_valuations
        all_valuations.sort(key=lambda v: v.value, reverse=True)
        ranked = [dataclasses.replace(v, rank=i) for i, v in enumerate(all_valuations, 1)]

        # 7. Persist
        if self._valuation_repo is not None:
            for v in ranked:
                self._valuation_repo.upsert(v)

        # 8. Build PredictResult
        predictions: list[dict[str, Any]] = []
        for v in ranked:
            predictions.append(
                {
                    "player_id": v.player_id,
                    "season": v.season,
                    "player_type": v.player_type,
                    "position": v.position,
                    "value": v.value,
                    "rank": v.rank,
                    "category_scores": v.category_scores,
                }
            )

        return PredictResult(
            model_name=self.name,
            predictions=predictions,
            output_path=config.output_dir or config.artifacts_dir,
        )

    def _value_pool(
        self,
        projections: list[Projection],
        categories: list[Any],
        position_map: dict[int, list[str]],
        league: LeagueSettings,
        budget: float,
        player_type: str,
        season: int,
        version: str,
        proj_system: str,
        *,
        pitcher_roster_spots: dict[str, int] | None = None,
        stdev_overrides: dict[str, float] | None = None,
        category_weights: dict[str, float] | None = None,
        use_direct_rates: bool = False,
        system: str = "zar",
        use_optimal_assignment: bool = True,
    ) -> list[Valuation]:
        """Run the full ZAR pipeline for one player pool (batters or pitchers)."""
        if not projections:
            return []

        stats_list = _extract_stats(projections)
        if pitcher_roster_spots is not None and "P" in pitcher_roster_spots:
            no_pos: list[str] = ["P"]
        elif league.roster_util > 0:
            no_pos = ["UTIL"]
        else:
            no_pos = []
        player_positions = [position_map.get(p.player_id, no_pos) for p in projections]
        roster_spots = build_roster_spots(league, pitcher_roster_spots=pitcher_roster_spots)

        result = run_zar_pipeline(
            stats_list,
            categories,
            player_positions,
            roster_spots,
            league.teams,
            budget,
            stdev_overrides=stdev_overrides,
            category_weights=category_weights,
            use_direct_rates=use_direct_rates,
            use_optimal_assignment=use_optimal_assignment,
        )

        # Build Valuation objects (rank=0 placeholder, filled later)
        valuations: list[Valuation] = []
        for i, proj in enumerate(projections):
            if result.assignments is not None and i in result.assignments:
                pos = result.assignments[i]
            else:
                pos = best_position(player_positions[i], result.replacement)
            valuations.append(
                Valuation(
                    player_id=proj.player_id,
                    season=season,
                    system=system,
                    version=version,
                    projection_system=proj_system,
                    projection_version=proj.version,
                    player_type=player_type,
                    position=pos,
                    value=round(result.dollar_values[i], 2),
                    rank=0,
                    category_scores=dict(result.z_scores[i].category_z),
                )
            )
        return valuations
