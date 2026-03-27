from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    CategoryConfig,
    CategoryNeed,
    Direction,
    LeagueSettings,
    PlayerRecommendation,
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


_PITCHER_POSITIONS = frozenset({"SP", "RP", "P"})


def _drafted_player_type(position: str) -> str:
    """Map a drafted position to 'batter' or 'pitcher'."""
    return "pitcher" if position in _PITCHER_POSITIONS else "batter"


def _classify_strength(rank: int, num_teams: int) -> str:
    """Classify rank into 'strong', 'average', or 'weak'."""
    third = num_teams / 3
    if rank <= third:
        return "strong"
    if rank > 2 * third:
        return "weak"
    return "average"


def _top_n_by_playing_time(
    players: list[Projection],
    n: int,
    playing_time_key: str,
) -> list[Projection]:
    """Return the top *n* players sorted by playing time (PA or IP).

    This approximates the draftable pool so that league averages reflect
    realistic per-team totals rather than the entire projection universe.
    """
    sorted_players = sorted(players, key=lambda p: p.stat_json.get(playing_time_key, 0.0), reverse=True)
    return sorted_players[:n]


def analyze_roster(
    player_ids: list[int],
    projections: list[Projection],
    league: LeagueSettings,
    roster_positions: dict[int, str] | None = None,
) -> RosterAnalysis:
    """Analyze a roster's projected category strengths and weaknesses.

    For each scoring category, sums counting stats or computes weighted-average
    rate stats across the roster, estimates a league rank, and classifies
    strength as 'strong', 'average', or 'weak'.

    *roster_positions* maps player_id → drafted position (e.g. "SS", "SP").
    When provided, only the projection matching the drafted role is used for
    each player — important for two-way players like Ohtani who can be drafted
    as batter or pitcher by different teams.
    """
    if not player_ids:
        return RosterAnalysis(
            projections=[],
            strongest_categories=[],
            weakest_categories=[],
        )

    roster_set = set(player_ids)
    num_teams = league.teams

    # Players can have both batter and pitcher projections, so collect all
    # matching projections rather than using a single-key dict.  Skip rows
    # with empty stat_json — these are stub entries (e.g. a pitcher with a
    # placeholder batter row) that would distort roster fill counts.
    # When roster_positions is provided, only keep the projection matching
    # the drafted role so two-way players contribute to the correct side.
    roster_projections = [p for p in projections if p.player_id in roster_set and p.stat_json]
    if roster_positions:
        roster_projections = [
            p
            for p in roster_projections
            if p.player_id not in roster_positions
            or p.player_type == _drafted_player_type(roster_positions[p.player_id])
        ]
    roster_batters = [p for p in roster_projections if p.player_type == "batter"]
    roster_pitchers = [p for p in roster_projections if p.player_type == "pitcher"]

    all_batters = _top_n_by_playing_time(
        [p for p in projections if p.player_type == "batter"],
        league.roster_batters * num_teams,
        "pa",
    )
    all_pitchers = _top_n_by_playing_time(
        [p for p in projections if p.player_type == "pitcher"],
        league.roster_pitchers * num_teams,
        "ip",
    )
    category_projections: list[TeamCategoryProjection] = []

    all_categories: list[tuple[CategoryConfig, list[Projection], list[Projection], int, int]] = [
        *(
            (cat, roster_batters, all_batters, len(roster_batters), league.roster_batters)
            for cat in league.batting_categories
        ),
        *(
            (cat, roster_pitchers, all_pitchers, len(roster_pitchers), league.roster_pitchers)
            for cat in league.pitching_categories
        ),
    ]

    for cat, roster_pool, league_pool, filled, total_slots in all_categories:
        team_value = _compute_category_value(cat, roster_pool)
        league_total = _compute_category_value(cat, league_pool)
        # For counting stats, divide total by teams to get per-team average,
        # then scale by roster fill fraction so partial rosters are compared
        # against proportional expectations (not a full-roster average).
        if cat.stat_type == StatType.COUNTING:
            league_avg = league_total / num_teams if num_teams > 0 else 0.0
            if total_slots > 0 and 0 < filled < total_slots:
                league_avg *= filled / total_slots
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


def compute_category_balance_scores(
    roster_ids: list[int],
    available_ids: list[int],
    projections: list[Projection],
    league: LeagueSettings,
) -> dict[int, float]:
    """Score available players by how much they address weak categories.

    Returns a dict mapping player_id → score in [0, 1], where 1.0 is the
    best category-balance improvement and 0.0 means no improvement.
    """
    if not available_ids:
        return {}

    analysis = analyze_roster(roster_ids, projections, league)
    weak_keys = {p.category for p in analysis.projections if p.strength == "weak"}
    if not weak_keys:
        return {pid: 0.0 for pid in available_ids}

    roster_set = set(roster_ids)
    roster_projections = [p for p in projections if p.player_id in roster_set]
    roster_batters = [p for p in roster_projections if p.player_type == "batter"]
    roster_pitchers = [p for p in roster_projections if p.player_type == "pitcher"]

    # Build per-type lookup for available players (prefer pitcher projection for
    # pitching categories and batter projection for batting categories).
    batter_lookup: dict[int, Projection] = {p.player_id: p for p in projections if p.player_type == "batter"}
    pitcher_lookup: dict[int, Projection] = {p.player_id: p for p in projections if p.player_type == "pitcher"}

    # Build list of (weak_cat_config, is_batting, roster_pool) tuples
    weak_cats: list[tuple[CategoryConfig, bool, list[Projection]]] = []
    for key in weak_keys:
        cat_config = _find_category_config(key, league)
        if cat_config is None:
            continue
        is_batting = _is_batting_category(key, league)
        pool = roster_batters if is_batting else roster_pitchers
        weak_cats.append((cat_config, is_batting, pool))

    # For each available player, sum positive impacts on matching-type weak categories
    raw_scores: dict[int, float] = {}
    for pid in available_ids:
        # Try both batter and pitcher projections for the candidate
        batter_candidate = batter_lookup.get(pid)
        pitcher_candidate = pitcher_lookup.get(pid)
        if batter_candidate is None and pitcher_candidate is None:
            raw_scores[pid] = 0.0
            continue
        total_impact = 0.0
        for cat_config, is_batting, roster_pool in weak_cats:
            candidate = batter_candidate if is_batting else pitcher_candidate
            if candidate is None:
                continue
            current_value = _compute_category_value(cat_config, roster_pool)
            with_player = _compute_category_value(cat_config, [*roster_pool, candidate])
            if cat_config.direction == Direction.LOWER:
                impact = current_value - with_player
            else:
                impact = with_player - current_value
            if impact > 0:
                total_impact += impact

        raw_scores[pid] = total_impact

    # Normalize to [0, 1]
    max_score = max(raw_scores.values()) if raw_scores else 0.0
    if max_score <= 0:
        return {pid: 0.0 for pid in available_ids}

    return {pid: raw / max_score for pid, raw in raw_scores.items()}


def _find_category_config(key: str, league: LeagueSettings) -> CategoryConfig | None:
    """Look up a CategoryConfig by key from league settings."""
    for cat in league.batting_categories:
        if cat.key == key:
            return cat
    for cat in league.pitching_categories:
        if cat.key == key:
            return cat
    return None


def _is_batting_category(key: str, league: LeagueSettings) -> bool:
    """Return True if key is a batting category."""
    return any(cat.key == key for cat in league.batting_categories)


def identify_needs(
    roster_ids: list[int],
    available_ids: list[int],
    projections: list[Projection],
    league: LeagueSettings,
    player_names: dict[int, str] | None = None,
    *,
    top_n: int = 5,
) -> list[CategoryNeed]:
    """Identify weak categories and recommend available players to address them.

    For each weak category, finds the available players that improve it the most
    and flags tradeoffs where improving one weak category hurts another.
    """
    analysis = analyze_roster(roster_ids, projections, league)
    names = player_names or {}
    num_teams = league.teams
    target_rank = int(2 * num_teams / 3)

    # Collect weak categories sorted by worst rank first
    weak_projections = [p for p in analysis.projections if p.strength == "weak"]
    weak_projections.sort(key=lambda p: p.league_rank_estimate, reverse=True)
    weak_keys = {p.category for p in weak_projections}

    roster_set = set(roster_ids)
    available_set = set(available_ids)
    roster_projections = [p for p in projections if p.player_id in roster_set]
    roster_batters = [p for p in roster_projections if p.player_type == "batter"]
    roster_pitchers = [p for p in roster_projections if p.player_type == "pitcher"]

    available_projections = [p for p in projections if p.player_id in available_set]
    available_batters = [p for p in available_projections if p.player_type == "batter"]
    available_pitchers = [p for p in available_projections if p.player_type == "pitcher"]

    result: list[CategoryNeed] = []

    for weak_proj in weak_projections:
        cat_config = _find_category_config(weak_proj.category, league)
        if cat_config is None:
            continue

        is_batting = _is_batting_category(weak_proj.category, league)
        roster_pool = roster_batters if is_batting else roster_pitchers
        available_pool = available_batters if is_batting else available_pitchers

        # Compute impact for each available player
        current_value = _compute_category_value(cat_config, roster_pool)
        candidates: list[tuple[Projection, float]] = []
        for candidate in available_pool:
            with_player = _compute_category_value(cat_config, [*roster_pool, candidate])
            if cat_config.direction == Direction.LOWER:
                impact = current_value - with_player  # lower is better → positive impact
            else:
                impact = with_player - current_value
            candidates.append((candidate, impact))

        # Sort by impact descending, take top_n
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_n]

        # Build recommendations with tradeoff detection
        # Identify other weak categories of the same type (batting/pitching)
        other_weak_same_type = [
            k for k in weak_keys if k != weak_proj.category and _is_batting_category(k, league) == is_batting
        ]

        recommendations: list[PlayerRecommendation] = []
        for candidate, impact in top_candidates:
            tradeoffs: list[str] = []
            for other_key in other_weak_same_type:
                other_config = _find_category_config(other_key, league)
                if other_config is None:
                    continue
                other_current = _compute_category_value(other_config, roster_pool)
                other_with = _compute_category_value(other_config, [*roster_pool, candidate])
                if other_config.direction == Direction.LOWER:
                    other_impact = other_current - other_with
                else:
                    other_impact = other_with - other_current
                if other_impact < 0:
                    tradeoffs.append(other_key)

            recommendations.append(
                PlayerRecommendation(
                    player_id=candidate.player_id,
                    player_name=names.get(candidate.player_id, ""),
                    player_type=candidate.player_type,
                    category_impact=impact,
                    tradeoff_categories=tuple(tradeoffs),
                )
            )

        result.append(
            CategoryNeed(
                category=weak_proj.category,
                current_rank=weak_proj.league_rank_estimate,
                target_rank=target_rank,
                best_available=tuple(recommendations),
            )
        )

    return result
