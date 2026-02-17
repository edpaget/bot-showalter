import dataclasses
from typing import Any

from fantasy_baseball_manager.domain.league_settings import LeagueSettings
from fantasy_baseball_manager.domain.model_run import ArtifactType
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.models.protocols import ModelConfig, PredictResult
from fantasy_baseball_manager.models.registry import register
from fantasy_baseball_manager.models.zar.engine import (
    compute_replacement_level,
    compute_var,
    compute_z_scores,
    convert_rate_stats,
    var_to_dollars,
)
from fantasy_baseball_manager.models.zar.positions import best_position, build_position_map
from fantasy_baseball_manager.repos.protocols import (
    PlayerRepo,
    PositionAppearanceRepo,
    ProjectionRepo,
    ValuationRepo,
)


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
        projection_repo: ProjectionRepo | None = None,
        player_repo: PlayerRepo | None = None,
        position_repo: PositionAppearanceRepo | None = None,
        valuation_repo: ValuationRepo | None = None,
    ) -> None:
        self._projection_repo = projection_repo
        self._player_repo = player_repo
        self._position_repo = position_repo
        self._valuation_repo = valuation_repo

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
        assert self._projection_repo is not None, "projection_repo is required for predict"
        assert self._position_repo is not None, "position_repo is required for predict"

        league: LeagueSettings = config.model_params["league"]
        proj_system: str = config.model_params["projection_system"]
        proj_version: str | None = config.model_params.get("projection_version")
        season = config.seasons[0] if config.seasons else 2025
        version = config.version or "1.0"

        # 1. Read projections and positions
        if proj_version is not None:
            projections = self._projection_repo.get_by_system_version(proj_system, proj_version)
            projections = [p for p in projections if p.season == season]
        else:
            projections = self._projection_repo.get_by_season(season, system=proj_system)
        appearances = self._position_repo.get_by_season(season)
        position_map = build_position_map(appearances, league)

        # 2. Split into batters and pitchers
        batter_projs = [p for p in projections if p.player_type == "batter"]
        pitcher_projs = [p for p in projections if p.player_type == "pitcher"]

        # 3. Budget split proportional to category count
        n_bat_cats = len(league.batting_categories)
        n_pitch_cats = len(league.pitching_categories)
        total_cats = n_bat_cats + n_pitch_cats
        total_league_budget = league.budget * league.teams
        batter_budget = total_league_budget * n_bat_cats / total_cats if total_cats else 0.0
        pitcher_budget = total_league_budget * n_pitch_cats / total_cats if total_cats else 0.0

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
        )

        # 5. Run ZAR for pitchers — all share a single "p" position
        pitcher_position_map: dict[int, list[str]] = {p.player_id: ["p"] for p in pitcher_projs}
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
            pitcher_roster_spots={"p": league.roster_pitchers},
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
    ) -> list[Valuation]:
        """Run the full ZAR pipeline for one player pool (batters or pitchers)."""
        if not projections:
            return []

        stats_list = _extract_stats(projections)
        category_keys = [c.key for c in categories]

        # Convert rate stats to marginal contributions
        converted = convert_rate_stats(stats_list, categories)

        # Compute z-scores
        z_scores = compute_z_scores(converted, category_keys)

        # Build per-player position lists (in same order as projections)
        player_positions = [position_map.get(p.player_id, ["util"]) for p in projections]

        # Roster spots
        if pitcher_roster_spots is not None:
            roster_spots = pitcher_roster_spots
        else:
            roster_spots = dict(league.positions)
            if league.roster_util > 0:
                roster_spots["util"] = league.roster_util

        # Replacement level and VAR
        replacement = compute_replacement_level(z_scores, player_positions, roster_spots, league.teams)
        var_values = compute_var(z_scores, replacement, player_positions)
        dollar_values = var_to_dollars(var_values, budget)

        # Build Valuation objects (rank=0 placeholder, filled later)
        valuations: list[Valuation] = []
        for i, proj in enumerate(projections):
            best_pos = best_position(player_positions[i], replacement)
            valuations.append(
                Valuation(
                    player_id=proj.player_id,
                    season=season,
                    system="zar",
                    version=version,
                    projection_system=proj_system,
                    projection_version=proj.version,
                    player_type=player_type,
                    position=best_pos,
                    value=round(dollar_values[i], 2),
                    rank=0,
                    category_scores=dict(z_scores[i].category_z),
                )
            )
        return valuations
