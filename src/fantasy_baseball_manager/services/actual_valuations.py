"""Compute ZAR valuations from end-of-season actual stats."""

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import Position, Valuation
from fantasy_baseball_manager.models.zar.engine import compute_budget_split, run_zar_pipeline
from fantasy_baseball_manager.models.zar.positions import best_position, build_position_map, build_roster_spots

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import EligibilityProvider, LeagueSettings
    from fantasy_baseball_manager.repos import (
        BattingStatsRepo,
        PitchingStatsRepo,
        PositionAppearanceRepo,
        ValuationRepo,
    )

logger = logging.getLogger(__name__)

_METADATA_FIELDS = frozenset({"id", "player_id", "season", "source", "team_id", "loaded_at"})


def _stats_to_dict(obj: object) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in dataclasses.fields(obj):  # type: ignore[arg-type]
        if field.name in _METADATA_FIELDS:
            continue
        value = getattr(obj, field.name)
        if isinstance(value, int | float):
            result[field.name] = float(value)
    return result


def compute_actual_valuations(
    season: int,
    league: LeagueSettings,
    batting_repo: BattingStatsRepo,
    pitching_repo: PitchingStatsRepo,
    position_repo: PositionAppearanceRepo,
    valuation_repo: ValuationRepo,
    *,
    actuals_source: str = "fangraphs",
    version: str = "1.0",
    eligibility_provider: EligibilityProvider | None = None,
) -> list[Valuation]:
    """Compute ZAR valuations from actual stats and persist them.

    Returns the ranked list of valuations (system='zar', projection_system='actual').
    """
    logger.info("Computing actual valuations for season %d", season)

    batting_actuals = batting_repo.get_by_season(season, source=actuals_source)
    pitching_actuals = pitching_repo.get_by_season(season, source=actuals_source)

    if not batting_actuals and not pitching_actuals:
        logger.warning("No actual stats for season %d", season)
        return []

    batter_stats = {bs.player_id: _stats_to_dict(bs) for bs in batting_actuals}
    pitcher_stats = {ps.player_id: _stats_to_dict(ps) for ps in pitching_actuals}

    appearances = position_repo.get_by_season(season)
    position_map = build_position_map(appearances, league)

    batter_budget, pitcher_budget = compute_budget_split(league)

    # Value batters
    batter_ids = [pid for pid in batter_stats if batter_stats[pid].get("pa", 0) > 0]
    batter_vals = _value_pool(
        player_ids=batter_ids,
        stats_map=batter_stats,
        categories=list(league.batting_categories),
        position_map=position_map,
        league=league,
        budget=batter_budget,
        player_type="batter",
        season=season,
        version=version,
    )

    # Value pitchers
    pitcher_ids = [pid for pid in pitcher_stats if pitcher_stats[pid].get("ip", 0) > 0]
    if league.pitcher_positions and eligibility_provider is not None:
        pitcher_position_map = eligibility_provider.get_pitcher_positions(season, league, pitcher_ids)
        pitcher_roster_spots = dict(league.pitcher_positions)
    else:
        pitcher_position_map: dict[int, list[str]] = {pid: [Position.P] for pid in pitcher_ids}
        pitcher_roster_spots: dict[str, int] = {Position.P: league.roster_pitchers}
    pitcher_vals = _value_pool(
        player_ids=pitcher_ids,
        stats_map=pitcher_stats,
        categories=list(league.pitching_categories),
        position_map=pitcher_position_map,
        league=league,
        budget=pitcher_budget,
        player_type="pitcher",
        season=season,
        version=version,
        pitcher_roster_spots=pitcher_roster_spots,
    )

    # Combine and rank by value descending
    all_valuations = batter_vals + pitcher_vals
    all_valuations.sort(key=lambda v: v.value, reverse=True)
    ranked = [dataclasses.replace(v, rank=i) for i, v in enumerate(all_valuations, 1)]

    # Persist
    for v in ranked:
        valuation_repo.upsert(v)

    logger.info("Computed %d actual valuations for season %d", len(ranked), season)
    return ranked


def _value_pool(
    player_ids: list[int],
    stats_map: dict[int, dict[str, float]],
    categories: list[Any],
    position_map: dict[int, list[str]],
    league: LeagueSettings,
    budget: float,
    player_type: str,
    season: int,
    version: str,
    *,
    pitcher_roster_spots: dict[str, int] | None = None,
) -> list[Valuation]:
    if not player_ids:
        return []

    stats_list = [stats_map[pid] for pid in player_ids]
    if pitcher_roster_spots is not None and Position.P in pitcher_roster_spots:
        no_pos: list[str] = [Position.P]
    elif league.roster_util > 0:
        no_pos = [Position.UTIL]
    else:
        no_pos = []
    player_positions = [position_map.get(pid, no_pos) for pid in player_ids]
    roster_spots = build_roster_spots(league, pitcher_roster_spots=pitcher_roster_spots)

    result = run_zar_pipeline(stats_list, categories, player_positions, roster_spots, league.teams, budget)

    valuations: list[Valuation] = []
    for i, pid in enumerate(player_ids):
        pos = best_position(player_positions[i], result.replacement)
        valuations.append(
            Valuation(
                player_id=pid,
                season=season,
                system="zar",
                version=version,
                projection_system="actual",
                projection_version=str(season),
                player_type=player_type,
                position=pos,
                value=round(result.dollar_values[i], 2),
                rank=0,
                category_scores=dict(result.z_scores[i].category_z),
            )
        )
    return valuations
