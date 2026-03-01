from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    CategoryConfig,
    Direction,
    LeagueSettings,
    RosterAnalysis,
    StatType,
    TeamCategoryProjection,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import Projection


def _resolve_numerator(expression: str, stats: dict[str, float]) -> float:
    """Sum compound numerator expressions like ``'bb+h'``."""
    parts = expression.split("+")
    return sum(stats.get(part.strip(), 0.0) for part in parts)


def _compute_category_value(
    category: CategoryConfig,
    roster_projections: list[Projection],
) -> float:
    """Compute the team-level value for a single category."""
    if category.stat_type == StatType.COUNTING:
        return sum(p.stat_json.get(category.key, 0.0) for p in roster_projections)

    # Rate stat: weighted by denominator
    total_numerator = 0.0
    total_denominator = 0.0
    for p in roster_projections:
        stats: dict[str, float] = p.stat_json
        if category.numerator is not None:
            total_numerator += _resolve_numerator(category.numerator, stats)
        if category.denominator is not None:
            total_denominator += stats.get(category.denominator, 0.0)

    if total_denominator == 0.0:
        return 0.0
    return total_numerator / total_denominator


def _estimate_league_rank(
    team_value: float,
    league_avg: float,
    num_teams: int,
    direction: Direction,
) -> int:
    """Estimate league rank (1 = best) using linear interpolation from league average."""
    if league_avg == 0.0:
        return (num_teams + 1) // 2

    ratio = team_value / league_avg

    # For LOWER-is-better stats, a lower ratio means better (lower rank number)
    deviation = 1.0 - ratio if direction == Direction.LOWER else ratio - 1.0

    # Linear interpolation: deviation of 0 → middle rank, positive → better rank
    # Scale factor: assume ~0.5 spread covers the full range of ranks
    mid = (num_teams + 1) / 2
    spread = 0.5
    rank = mid - deviation / spread * (mid - 1)
    return max(1, min(num_teams, round(rank)))


def _classify_strength(rank: int, num_teams: int) -> str:
    """Classify rank into 'strong', 'average', or 'weak'."""
    third = num_teams / 3
    if rank <= third:
        return "strong"
    if rank > 2 * third:
        return "weak"
    return "average"


def analyze_roster(
    player_ids: list[int],
    projections: list[Projection],
    league: LeagueSettings,
) -> RosterAnalysis:
    """Analyze a roster's projected category strengths and weaknesses.

    For each scoring category, sums counting stats or computes weighted-average
    rate stats across the roster, estimates a league rank, and classifies
    strength as 'strong', 'average', or 'weak'.
    """
    if not player_ids:
        return RosterAnalysis(
            projections=[],
            strongest_categories=[],
            weakest_categories=[],
        )

    roster_set = set(player_ids)
    proj_lookup: dict[int, Projection] = {p.player_id: p for p in projections}

    roster_projections = [proj_lookup[pid] for pid in roster_set if pid in proj_lookup]
    roster_batters = [p for p in roster_projections if p.player_type == "batter"]
    roster_pitchers = [p for p in roster_projections if p.player_type == "pitcher"]

    all_batters = [p for p in projections if p.player_type == "batter"]
    all_pitchers = [p for p in projections if p.player_type == "pitcher"]

    num_teams = league.teams
    category_projections: list[TeamCategoryProjection] = []

    all_categories: list[tuple[CategoryConfig, list[Projection], list[Projection]]] = [
        *((cat, roster_batters, all_batters) for cat in league.batting_categories),
        *((cat, roster_pitchers, all_pitchers) for cat in league.pitching_categories),
    ]

    for cat, roster_pool, league_pool in all_categories:
        team_value = _compute_category_value(cat, roster_pool)
        league_total = _compute_category_value(cat, league_pool)
        # For counting stats, divide total by teams to get per-team average.
        # For rate stats, the pool-wide rate IS the expected per-team rate.
        if cat.stat_type == StatType.COUNTING:
            league_avg = league_total / num_teams if num_teams > 0 else 0.0
        else:
            league_avg = league_total

        rank = _estimate_league_rank(team_value, league_avg, num_teams, cat.direction)
        strength = _classify_strength(rank, num_teams)

        category_projections.append(
            TeamCategoryProjection(
                category=cat.key,
                projected_value=team_value,
                league_rank_estimate=rank,
                strength=strength,
            )
        )

    strongest = [p.category for p in category_projections if p.strength == "strong"]
    weakest = [p.category for p in category_projections if p.strength == "weak"]

    return RosterAnalysis(
        projections=category_projections,
        strongest_categories=strongest,
        weakest_categories=weakest,
    )
