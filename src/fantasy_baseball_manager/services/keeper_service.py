from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    AdjustedValuation,
    Err,
    KeeperCost,
    KeeperDecision,
    Ok,
    Roster,
    TradeEvaluation,
    TradePlayerDetail,
)
from fantasy_baseball_manager.models.zar.engine import compute_budget_split, run_zar_pipeline
from fantasy_baseball_manager.models.zar.positions import best_position, build_roster_spots

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
    matches = player_repo.search_by_name(player_name)

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
    # Build lookup: player_id -> highest valuation
    val_lookup: dict[int, Valuation] = {}
    for v in valuations:
        existing = val_lookup.get(v.player_id)
        if existing is None or v.value > existing.value:
            val_lookup[v.player_id] = v

    # Build lookup: player_id -> Player
    player_lookup: dict[int, Player] = {}
    for p in players:
        if p.id is not None:
            player_lookup[p.id] = p

    decisions: list[KeeperDecision] = []
    for kc in keeper_costs:
        val = val_lookup.get(kc.player_id)
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
    val_lookup: dict[int, float] = {}
    for v in valuations:
        existing = val_lookup.get(v.player_id)
        if existing is None or v.value > existing:
            val_lookup[v.player_id] = v.value

    keeper_ids: set[int] = set()
    for roster in rosters:
        candidates: list[tuple[int, float]] = []
        for entry in roster.entries:
            if entry.player_id is None:
                continue
            value = val_lookup.get(entry.player_id, 0.0)
            candidates.append((entry.player_id, value))
        candidates.sort(key=lambda x: x[1], reverse=True)
        for pid, _ in candidates[:max_keepers]:
            keeper_ids.add(pid)

    return keeper_ids


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
    val_lookup: dict[int, Valuation] = {}
    for v in valuations:
        existing = val_lookup.get(v.player_id)
        if existing is None or v.value > existing.value:
            val_lookup[v.player_id] = v

    cost_lookup: dict[int, KeeperCost] = {}
    for kc in keeper_costs:
        cost_lookup[kc.player_id] = kc

    player_lookup: dict[int, Player] = {}
    for p in players:
        if p.id is not None:
            player_lookup[p.id] = p

    def _build_detail(player_id: int) -> TradePlayerDetail:
        val = val_lookup.get(player_id)
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


def compute_adjusted_valuations(
    kept_player_ids: set[int],
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
    """
    # Filter out kept players
    pool = [p for p in projections if p.player_id not in kept_player_ids]

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

    # Build lookups
    orig_lookup: dict[int, float] = {}
    for v in original_valuations:
        existing = orig_lookup.get(v.player_id)
        if existing is None or v.value > existing:
            orig_lookup[v.player_id] = v.value

    player_lookup: dict[int, Player] = {}
    for p in players:
        if p.id is not None:
            player_lookup[p.id] = p

    # Build results
    results: list[AdjustedValuation] = []
    for player_id, adjusted_value, player_type, position in all_vals:
        original_value = orig_lookup.get(player_id, 0.0)
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
