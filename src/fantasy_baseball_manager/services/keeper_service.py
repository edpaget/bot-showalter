from dataclasses import replace
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    AdjustedValuation,
    CategoryNeed,
    Err,
    KeeperCost,
    KeeperDecision,
    LeagueKeeperOverview,
    Ok,
    ProjectedKeeper,
    Roster,
    RosterAnalysis,
    TeamKeeperProjection,
    TradeEvaluation,
    TradePlayerDetail,
    TradeTarget,
)
from fantasy_baseball_manager.models.zar.engine import compute_budget_split, run_zar_pipeline
from fantasy_baseball_manager.models.zar.positions import best_position, build_roster_spots
from fantasy_baseball_manager.name_utils import resolve_players
from fantasy_baseball_manager.services.category_tracker import analyze_roster, identify_needs

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        CategoryConfig,
        LeagueSettings,
        Player,
        Projection,
        Result,
        Valuation,
    )
    from fantasy_baseball_manager.repos import KeeperCostRepo, PlayerRepo


def _best_valuation_for_player(
    player_id: int,
    val_lookup: dict[tuple[int, str], Valuation],
    player_type: str | None = None,
) -> Valuation | None:
    """Find the best matching valuation for a player from a typed lookup.

    If player_type is given, returns the exact match. Otherwise returns
    the highest-value valuation across all types for that player.
    """
    if player_type is not None:
        return val_lookup.get((player_id, player_type))
    best: Valuation | None = None
    for (pid, _ptype), v in val_lookup.items():
        if pid == player_id and (best is None or v.value > best.value):
            best = v
    return best


def _best_value_for_player(
    player_id: int,
    val_lookup: dict[tuple[int, str], float],
) -> float:
    """Find the highest value across all types for a player from a typed lookup."""
    best = 0.0
    for (pid, _), v in val_lookup.items():
        if pid == player_id and v > best:
            best = v
    return best


def set_keeper_cost(
    player_name: str,
    cost: float,
    season: int,
    league: str,
    player_repo: PlayerRepo,
    keeper_repo: KeeperCostRepo,
    years_remaining: int = 1,
    source: str = "auction",
    original_round: int | None = None,
) -> Result[KeeperCost, str]:
    """Resolve a player by name and set their keeper cost.

    Returns Ok(KeeperCost) on success, or Err(message) if the player
    cannot be uniquely resolved.
    """
    matches = resolve_players(player_repo, player_name)

    if len(matches) == 0:
        return Err(f"no player found matching '{player_name}'")

    if len(matches) > 1:
        names = [f"{p.name_first} {p.name_last}" for p in matches]
        return Err(f"ambiguous name '{player_name}', matches: {', '.join(names)}")

    player = matches[0]
    assert player.id is not None  # noqa: S101

    keeper_cost = KeeperCost(
        player_id=player.id,
        season=season,
        league=league,
        cost=cost,
        years_remaining=years_remaining,
        source=source,
        original_round=original_round,
    )
    keeper_repo.upsert_batch([keeper_cost])
    return Ok(keeper_cost)


def compute_surplus(
    keeper_costs: list[KeeperCost],
    valuations: list[Valuation],
    players: list[Player],
    threshold: float = 0.0,
    decay: float = 0.85,
) -> list[KeeperDecision]:
    """Compute surplus value for each keeper cost and recommend keep/release.

    Takes pre-fetched data (not repos) — keeps the function pure and easy to test.
    For players with multiple valuations, the highest value is used.
    Multi-year contracts use discounted future surplus: sum(surplus * decay^i).
    """
    # Build lookup: (player_id, player_type) -> highest valuation
    val_lookup: dict[tuple[int, str], Valuation] = {}
    for v in valuations:
        key = (v.player_id, v.player_type)
        existing = val_lookup.get(key)
        if existing is None or v.value > existing.value:
            val_lookup[key] = v

    # Build lookup: player_id -> Player
    player_lookup: dict[int, Player] = {}
    for p in players:
        if p.id is not None:
            player_lookup[p.id] = p

    decisions: list[KeeperDecision] = []
    for kc in keeper_costs:
        val = _best_valuation_for_player(kc.player_id, val_lookup, kc.player_type)
        projected_value = val.value if val is not None else 0.0
        position = val.position if val is not None else "UTIL"

        single_year_surplus = projected_value - kc.cost
        total_surplus = sum(single_year_surplus * decay**i for i in range(kc.years_remaining))

        player = player_lookup.get(kc.player_id)
        player_name = f"{player.name_first} {player.name_last}" if player is not None else "Unknown"

        recommendation = "keep" if total_surplus >= threshold else "release"

        decisions.append(
            KeeperDecision(
                player_id=kc.player_id,
                player_name=player_name,
                player_type=val.player_type if val is not None else "",
                position=position,
                cost=kc.cost,
                projected_value=projected_value,
                surplus=total_surplus,
                years_remaining=kc.years_remaining,
                recommendation=recommendation,
                original_round=kc.original_round,
            )
        )

    decisions.sort(key=lambda d: d.surplus, reverse=True)
    return decisions


def estimate_other_keepers(
    rosters: list[Roster],
    valuations: list[Valuation],
    max_keepers: int,
) -> set[int]:
    """Estimate which players other teams will keep based on projected value.

    For each roster, takes the top `max_keepers` players by valuation value.
    Returns the union of all estimated keeper player IDs.
    """
    val_lookup: dict[tuple[int, str], float] = {}
    for v in valuations:
        key = (v.player_id, v.player_type)
        existing = val_lookup.get(key)
        if existing is None or v.value > existing:
            val_lookup[key] = v.value

    keeper_ids: set[int] = set()
    for roster in rosters:
        candidates: list[tuple[int, float]] = []
        for entry in roster.entries:
            if entry.player_id is None:
                continue
            value = _best_value_for_player(entry.player_id, val_lookup)
            candidates.append((entry.player_id, value))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for pid, _ in candidates[:max_keepers]:
            keeper_ids.add(pid)

    return keeper_ids


def build_league_keeper_overview(
    rosters: list[Roster],
    valuations: list[Valuation],
    players: list[Player],
    max_keepers: int,
    user_team_key: str,
    team_names: dict[str, str],
) -> LeagueKeeperOverview:
    """Build a league-wide keeper overview with per-team projections and trade targets."""
    # Build lookups
    val_lookup: dict[tuple[int, str], Valuation] = {}
    for v in valuations:
        key = (v.player_id, v.player_type)
        existing = val_lookup.get(key)
        if existing is None or v.value > existing.value:
            val_lookup[key] = v

    player_lookup: dict[int, Player] = {}
    for p in players:
        if p.id is not None:
            player_lookup[p.id] = p

    all_category_names: set[str] = set()
    team_projections: list[TeamKeeperProjection] = []
    # Track per-team ranked candidates for trade target identification
    team_ranked: dict[str, list[tuple[int, float, str, str, dict[str, float]]]] = {}

    for roster in rosters:
        candidates: list[tuple[int, float, str, str, dict[str, float]]] = []
        for entry in roster.entries:
            if entry.player_id is None:
                continue
            val = _best_valuation_for_player(entry.player_id, val_lookup)
            value = val.value if val is not None else 0.0
            position = val.position if val is not None else "UTIL"
            cat_scores = val.category_scores if val is not None else {}
            player = player_lookup.get(entry.player_id)
            name = f"{player.name_first} {player.name_last}" if player is not None else entry.player_name
            candidates.append((entry.player_id, value, position, name, cat_scores))

        candidates.sort(key=lambda x: x[1], reverse=True)
        team_ranked[roster.team_key] = candidates

        keeper_candidates = candidates[:max_keepers]
        keepers = tuple(
            ProjectedKeeper(
                player_id=pid,
                player_name=name,
                position=pos,
                value=val,
                category_scores=cats,
            )
            for pid, val, pos, name, cats in keeper_candidates
        )

        category_totals: dict[str, float] = {}
        for keeper in keepers:
            for cat, score in keeper.category_scores.items():
                category_totals[cat] = category_totals.get(cat, 0.0) + score
            all_category_names.update(keeper.category_scores.keys())

        total_value = sum(k.value for k in keepers)
        team_projections.append(
            TeamKeeperProjection(
                team_key=roster.team_key,
                team_name=team_names.get(roster.team_key, roster.team_key),
                is_user=(roster.team_key == user_team_key),
                keepers=keepers,
                total_value=total_value,
                category_totals=category_totals,
            )
        )

    team_projections.sort(key=lambda t: t.total_value, reverse=True)

    # Find user's worst keeper value
    user_proj = next((tp for tp in team_projections if tp.is_user), None)
    user_worst = min((k.value for k in user_proj.keepers), default=0.0) if user_proj else 0.0

    # Build trade targets: non-user surplus players above user's worst keeper
    trade_targets: list[TradeTarget] = []
    for roster in rosters:
        if roster.team_key == user_team_key:
            continue
        ranked = team_ranked.get(roster.team_key, [])
        for rank_idx, (pid, value, pos, name, _cats) in enumerate(ranked, 1):
            if rank_idx <= max_keepers:
                continue
            if value > user_worst:
                trade_targets.append(
                    TradeTarget(
                        player_id=pid,
                        player_name=name,
                        position=pos,
                        value=value,
                        owning_team_name=team_names.get(roster.team_key, roster.team_key),
                        owning_team_key=roster.team_key,
                        rank_on_team=rank_idx,
                    )
                )

    trade_targets.sort(key=lambda t: t.value, reverse=True)

    return LeagueKeeperOverview(
        team_projections=tuple(team_projections),
        trade_targets=tuple(trade_targets),
        category_names=tuple(sorted(all_category_names)),
    )


def build_keeper_draft_needs(
    rosters: list[Roster],
    valuations: list[Valuation],
    players: list[Player],
    projections: list[Projection],
    max_keepers: int,
    user_team_key: str,
    team_names: dict[str, str],
    league: LeagueSettings,
    *,
    top_n: int = 5,
) -> tuple[RosterAnalysis, list[CategoryNeed]]:
    """Combine keeper projection with category needs analysis.

    Returns the user's roster analysis (strengths/weaknesses from keepers)
    and a list of category needs with recommended players from the post-keeper pool.
    """
    overview = build_league_keeper_overview(
        rosters=rosters,
        valuations=valuations,
        players=players,
        max_keepers=max_keepers,
        user_team_key=user_team_key,
        team_names=team_names,
    )

    # Find user's projected keepers
    user_proj = next((tp for tp in overview.team_projections if tp.is_user), None)
    user_keeper_ids = [k.player_id for k in user_proj.keepers] if user_proj else []

    # Get all estimated keeper IDs (all teams)
    all_keeper_ids = estimate_other_keepers(rosters, valuations, max_keepers)

    # Available pool = players with projections minus all keepers
    proj_player_ids = {p.player_id for p in projections}
    available_ids = sorted(proj_player_ids - all_keeper_ids)

    # Build player name lookup for recommendations
    player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

    # Analyze the user's keeper roster
    analysis = analyze_roster(user_keeper_ids, projections, league)

    # Identify needs
    needs = identify_needs(
        roster_ids=user_keeper_ids,
        available_ids=available_ids,
        projections=projections,
        league=league,
        player_names=player_names,
        top_n=top_n,
    )

    return analysis, needs


def adjust_valuations_for_league_keepers(
    rosters: list[Roster],
    valuations: list[Valuation],
    projections: list[Projection],
    batter_positions: dict[int, list[str]],
    pitcher_positions: dict[int, list[str]],
    league: LeagueSettings,
    players: list[Player],
    max_keepers: int,
) -> list[Valuation]:
    """Estimate other teams' keepers and return valuations adjusted for draft pool depletion.

    Combines keeper estimation with ZAR revaluation: estimates which players
    other teams will keep, removes them from the draft pool, re-runs valuations,
    and returns a new valuation list with adjusted dollar values.

    Returns the original valuations unchanged if no keepers are estimated.
    """
    estimated_ids = estimate_other_keepers(rosters, valuations, max_keepers)
    if not estimated_ids:
        return valuations

    adjusted = compute_adjusted_valuations(
        {(pid, None) for pid in estimated_ids},
        projections,
        batter_positions,
        pitcher_positions,
        league,
        valuations,
        players,
    )

    adj_lookup = {a.player_id: a for a in adjusted}
    result: list[Valuation] = []
    for v in valuations:
        adj = adj_lookup.get(v.player_id)
        if adj is not None:
            result.append(replace(v, value=adj.adjusted_value))
        elif v.player_id not in estimated_ids:
            result.append(v)

    return result


def evaluate_trade(
    team_a_gives: list[int],
    team_b_gives: list[int],
    keeper_costs: list[KeeperCost],
    valuations: list[Valuation],
    players: list[Player],
    decay: float = 0.85,
) -> TradeEvaluation:
    """Evaluate a trade between two teams using surplus value.

    Computes net surplus exchanged by each side.  Team A "gives" the players
    in *team_a_gives* and "receives" those in *team_b_gives* (and vice-versa).
    """
    # Build lookups
    val_lookup: dict[tuple[int, str], Valuation] = {}
    for v in valuations:
        key = (v.player_id, v.player_type)
        existing = val_lookup.get(key)
        if existing is None or v.value > existing.value:
            val_lookup[key] = v

    cost_lookup: dict[int, KeeperCost] = {}
    for kc in keeper_costs:
        cost_lookup[kc.player_id] = kc

    player_lookup: dict[int, Player] = {}
    for p in players:
        if p.id is not None:
            player_lookup[p.id] = p

    def _build_detail(player_id: int) -> TradePlayerDetail:
        val = _best_valuation_for_player(player_id, val_lookup)
        projected_value = val.value if val is not None else 0.0
        position = val.position if val is not None else "UTIL"

        kc = cost_lookup.get(player_id)
        cost = kc.cost if kc is not None else 0.0
        years = kc.years_remaining if kc is not None else 1

        single_year_surplus = projected_value - cost
        surplus = sum(single_year_surplus * decay**i for i in range(years))

        player = player_lookup.get(player_id)
        name = f"{player.name_first} {player.name_last}" if player is not None else "Unknown"

        return TradePlayerDetail(
            player_id=player_id,
            player_name=name,
            position=position,
            cost=cost,
            projected_value=projected_value,
            surplus=surplus,
            years_remaining=years,
        )

    a_details = [_build_detail(pid) for pid in team_a_gives]
    b_details = [_build_detail(pid) for pid in team_b_gives]

    a_gives_surplus = sum(d.surplus for d in a_details)
    b_gives_surplus = sum(d.surplus for d in b_details)

    team_a_delta = b_gives_surplus - a_gives_surplus
    team_b_delta = -team_a_delta

    if team_a_delta > 0:
        winner = "team_a"
    elif team_a_delta < 0:
        winner = "team_b"
    else:
        winner = "even"

    return TradeEvaluation(
        team_a_gives=a_details,
        team_b_gives=b_details,
        team_a_surplus_delta=team_a_delta,
        team_b_surplus_delta=team_b_delta,
        winner=winner,
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


def _is_projection_kept(player_id: int, player_type: str, kept_keys: set[tuple[int, str | None]]) -> bool:
    """Check if a projection matches a kept key.

    A ``None`` player_type in the key matches all types for that player
    (backward compat for untyped keepers).
    """
    return any(pid == player_id and (ptype is None or ptype == player_type) for pid, ptype in kept_keys)


def compute_adjusted_valuations(
    kept_player_ids: set[tuple[int, str | None]],
    projections: list[Projection],
    batter_positions: dict[int, list[str]],
    pitcher_positions: dict[int, list[str]],
    league: LeagueSettings,
    original_valuations: list[Valuation],
    players: list[Player],
) -> list[AdjustedValuation]:
    """Re-run ZAR valuations on the post-keeper draft pool.

    Filters kept players out before running the pipeline, so replacement
    levels naturally recalculate based on the reduced pool.

    ``kept_player_ids`` is a set of ``(player_id, player_type | None)`` tuples.
    A ``None`` player_type matches all types for that player.
    """
    # Filter out kept players
    pool = [p for p in projections if not _is_projection_kept(p.player_id, p.player_type, kept_player_ids)]

    # Split into batters and pitchers
    batter_projs = [p for p in pool if p.player_type == "batter" and p.stat_json.get("pa", 0) > 0]
    pitcher_projs = [p for p in pool if p.player_type == "pitcher" and p.stat_json.get("ip", 0) > 0]

    bat_budget, pit_budget = compute_budget_split(league)

    # Value batters
    batter_vals = _value_pool(batter_projs, list(league.batting_categories), batter_positions, league, bat_budget)

    # Value pitchers
    pitcher_roster = league.pitcher_positions or {"p": league.roster_pitchers}
    pitcher_vals = _value_pool(
        pitcher_projs,
        list(league.pitching_categories),
        pitcher_positions,
        league,
        pit_budget,
        pitcher_roster_spots=pitcher_roster,
    )

    # Combine and sort
    all_vals = batter_vals + pitcher_vals
    all_vals.sort(key=lambda v: v[1], reverse=True)  # sort by dollar value

    # Build lookups — keyed by (player_id, player_type) to avoid collisions
    # for two-way players who have both batter and pitcher valuations.
    orig_lookup: dict[tuple[int, str], float] = {}
    for v in original_valuations:
        key = (v.player_id, v.player_type)
        existing = orig_lookup.get(key)
        if existing is None or v.value > existing:
            orig_lookup[key] = v.value

    player_lookup: dict[int, Player] = {}
    for p in players:
        if p.id is not None:
            player_lookup[p.id] = p

    # Build results
    results: list[AdjustedValuation] = []
    for player_id, adjusted_value, player_type, position in all_vals:
        original_value = orig_lookup.get((player_id, player_type), 0.0)
        player = player_lookup.get(player_id)
        player_name = f"{player.name_first} {player.name_last}" if player else "Unknown"
        results.append(
            AdjustedValuation(
                player_id=player_id,
                player_name=player_name,
                player_type=player_type,
                position=position,
                original_value=round(original_value, 2),
                adjusted_value=round(adjusted_value, 2),
                value_change=round(adjusted_value - original_value, 2),
            )
        )

    return results


def _value_pool(
    projections: list[Projection],
    categories: list[CategoryConfig],
    position_map: dict[int, list[str]],
    league: LeagueSettings,
    budget: float,
    *,
    pitcher_roster_spots: dict[str, int] | None = None,
) -> list[tuple[int, float, str, str]]:
    """Run ZAR pipeline on a player pool, return (player_id, dollars, player_type, position)."""
    if not projections:
        return []

    stats_list = _extract_stats(projections)

    if pitcher_roster_spots is not None and "p" in pitcher_roster_spots:
        no_pos: list[str] = ["p"]
    elif league.roster_util > 0:
        no_pos = ["util"]
    else:
        no_pos = []

    player_positions = [position_map.get(p.player_id, no_pos) for p in projections]
    roster_spots = build_roster_spots(league, pitcher_roster_spots=pitcher_roster_spots)

    result = run_zar_pipeline(stats_list, categories, player_positions, roster_spots, league.teams, budget)

    valuations: list[tuple[int, float, str, str]] = []
    for i, proj in enumerate(projections):
        pos = best_position(player_positions[i], result.replacement)
        valuations.append((proj.player_id, round(result.dollar_values[i], 2), proj.player_type, pos))

    return valuations
