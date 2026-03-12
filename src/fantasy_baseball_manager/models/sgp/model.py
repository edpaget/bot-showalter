import dataclasses
from typing import TYPE_CHECKING, Any, Protocol

from fantasy_baseball_manager.domain import (
    ArtifactType,
    EligibilityProvider,
    Position,
    SgpDenominators,
    StatType,
    Valuation,
)
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.models.sgp.engine import run_sgp_pipeline
from fantasy_baseball_manager.models.zar.engine import compute_budget_split
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


class DenominatorProvider(Protocol):
    """Compute SGP denominators for a league's categories."""

    def __call__(self, league: LeagueSettings, *, method: str | None = None) -> SgpDenominators: ...


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


@register("sgp")
class SgpModel:
    def __init__(
        self,
        projection_repo: ProjectionRepo,
        position_repo: PositionAppearanceRepo,
        player_repo: PlayerRepo | None = None,
        valuation_repo: ValuationRepo | None = None,
        eligibility_service: EligibilityProvider | None = None,
        denominator_provider: DenominatorProvider | None = None,
    ) -> None:
        self._projection_repo = projection_repo
        self._player_repo = player_repo
        self._position_repo = position_repo
        self._valuation_repo = valuation_repo
        self._eligibility_service = eligibility_service
        self._denominator_provider = denominator_provider

    @property
    def name(self) -> str:
        return "sgp"

    @property
    def description(self) -> str:
        return "Standings Gain Points valuation for H2H categories leagues"

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
        season = config.seasons[0] if config.seasons else 2025
        version = config.version or "1.0"

        # 1. Get SGP denominators (pre-computed or from DB via provider)
        denominators_input: SgpDenominators | dict[str, float] | None = config.model_params.get("denominators")
        representative_team: dict[str, tuple[float, float]] = {}
        if denominators_input is None:
            if self._denominator_provider is None:
                msg = "denominators must be provided in model_params or via denominator_provider"
                raise TypeError(msg)
            denominator_method: str | None = config.model_params.get("denominator_method")
            sgp_denoms = self._denominator_provider(league, method=denominator_method)
            denominators = sgp_denoms.averages
            representative_team = sgp_denoms.representative_team
        elif isinstance(denominators_input, dict):
            denominators = denominators_input
        else:
            denominators = denominators_input.averages
            representative_team = denominators_input.representative_team
        # Allow explicit override from model_params
        if "representative_team" in config.model_params:
            representative_team = config.model_params["representative_team"]

        # 2. Read projections
        projections: list[Projection] = config.model_params.get("projections") or []
        if not projections:
            if proj_version is not None:
                projections = self._projection_repo.get_by_system_version(proj_system, proj_version)
                projections = [p for p in projections if p.season == season]
            else:
                projections = self._projection_repo.get_by_season(season, system=proj_system)
        position_map = self._eligibility_service.get_batter_positions(season, league)

        # 3. Split into batters and pitchers
        min_pa = league.eligibility.min_pa or 1
        min_ip = league.eligibility.min_ip or 1
        batter_projs = [p for p in projections if p.player_type == "batter" and p.stat_json.get("pa", 0) >= min_pa]
        pitcher_projs = [p for p in projections if p.player_type == "pitcher" and p.stat_json.get("ip", 0) >= min_ip]

        # 3.5. Direct rate stats, volume weighting, and category weights flags
        use_direct_rates: bool = config.model_params.get("use_direct_rates", False)
        use_optimal_assignment: bool = config.model_params.get("use_optimal_assignment", True)
        volume_weighted: bool = config.model_params.get("volume_weighted", False)
        category_weights: dict[str, float] | None = config.model_params.get("category_weights")

        # 4. Budget split
        batter_budget, pitcher_budget = compute_budget_split(league)

        # 4.5. Compute projection-based fallback for categories missing from standings
        all_categories = list(league.batting_categories) + list(league.pitching_categories)
        batter_stats = _extract_stats(batter_projs)
        pitcher_stats = _extract_stats(pitcher_projs)
        for cat in all_categories:
            if cat.stat_type is StatType.RATE and cat.denominator and cat.key not in representative_team:
                # Derive avg_volume from projections for this category
                all_stats = batter_stats + pitcher_stats
                vols = [s.get(cat.denominator, 0.0) for s in all_stats if s.get(cat.denominator, 0.0) > 0]
                rates = [
                    s.get(cat.key, 0.0)
                    for s in all_stats
                    if s.get(cat.key) is not None and s.get(cat.denominator, 0.0) > 0
                ]
                if vols and rates and league.teams > 0:
                    avg_vol = sum(vols) / league.teams
                    avg_rate = sum(r * v for r, v in zip(rates, vols, strict=True)) / sum(vols)
                    representative_team[cat.key] = (avg_rate, avg_vol)

        # 5. Run SGP for batters
        batter_valuations = self._value_pool(
            batter_projs,
            list(league.batting_categories),
            denominators,
            position_map,
            league,
            batter_budget,
            "batter",
            season,
            version,
            proj_system,
            use_direct_rates=use_direct_rates,
            use_optimal_assignment=use_optimal_assignment,
            volume_weighted=volume_weighted,
            representative_team=representative_team or None,
            category_weights=category_weights,
        )

        # 6. Run SGP for pitchers
        pitcher_position_map = self._eligibility_service.get_pitcher_positions(
            season, league, [p.player_id for p in pitcher_projs]
        )
        if league.pitcher_positions:
            pitcher_roster_spots = dict(league.pitcher_positions)
        else:
            pitcher_roster_spots = {Position.P: league.roster_pitchers}
        pitcher_valuations = self._value_pool(
            pitcher_projs,
            list(league.pitching_categories),
            denominators,
            pitcher_position_map,
            league,
            pitcher_budget,
            "pitcher",
            season,
            version,
            proj_system,
            pitcher_roster_spots=pitcher_roster_spots,
            use_direct_rates=use_direct_rates,
            use_optimal_assignment=use_optimal_assignment,
            volume_weighted=volume_weighted,
            representative_team=representative_team or None,
            category_weights=category_weights,
        )

        # 7. Rank all valuations combined
        all_valuations = batter_valuations + pitcher_valuations
        all_valuations.sort(key=lambda v: v.value, reverse=True)
        ranked = [dataclasses.replace(v, rank=i) for i, v in enumerate(all_valuations, 1)]

        # 8. Persist
        if self._valuation_repo is not None:
            for v in ranked:
                self._valuation_repo.upsert(v)

        # 9. Build PredictResult
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
        denominators: dict[str, float],
        position_map: dict[int, list[str]],
        league: LeagueSettings,
        budget: float,
        player_type: str,
        season: int,
        version: str,
        proj_system: str,
        *,
        pitcher_roster_spots: dict[str, int] | None = None,
        use_direct_rates: bool = False,
        use_optimal_assignment: bool = True,
        volume_weighted: bool = False,
        representative_team: dict[str, tuple[float, float]] | None = None,
        category_weights: dict[str, float] | None = None,
    ) -> list[Valuation]:
        """Run the full SGP pipeline for one player pool."""
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

        result = run_sgp_pipeline(
            stats_list,
            categories,
            denominators,
            player_positions,
            roster_spots,
            league.teams,
            budget,
            use_direct_rates=use_direct_rates,
            use_optimal_assignment=use_optimal_assignment,
            volume_weighted=volume_weighted,
            representative_team=representative_team,
            category_weights=category_weights,
        )

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
                    system="sgp",
                    version=version,
                    projection_system=proj_system,
                    projection_version=proj.version,
                    player_type=player_type,
                    position=pos,
                    value=round(result.dollar_values[i], 2),
                    rank=0,
                    category_scores=dict(result.sgp_scores[i].category_sgp),
                )
            )
        return valuations
