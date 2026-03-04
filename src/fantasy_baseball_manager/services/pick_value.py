from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import (
    CascadeResult,
    CascadeRoster,
    DraftPick,
    KeeperCost,
    PickTrade,
    PickTradeEvaluation,
    PickValue,
    PickValueCurve,
)
from fantasy_baseball_manager.services.draft_state import build_draft_roster_slots

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        ADP,
        DraftBoard,
        DraftBoardRow,
        LeagueSettings,
        Valuation,
    )

_SMOOTHING_WINDOW = 5
_DRAFT_ROUND_FLOOR_COST = 1.0


def value_at(curve: PickValueCurve, pick: int) -> float:
    """Return expected value for a pick number, 0.0 if out of range."""
    for pv in curve.picks:
        if pv.pick == pick:
            return pv.expected_value
    return 0.0


def round_to_dollar_cost(
    round_num: int,
    league: LeagueSettings,
    curve: PickValueCurve,
) -> float:
    """Convert a draft round to a dollar cost using the pick value curve.

    Averages the expected value of all picks in the round that have positive
    values on the curve.  Returns at least ``_DRAFT_ROUND_FLOOR_COST``.
    """
    start = (round_num - 1) * league.teams + 1
    end = round_num * league.teams
    values = [value_at(curve, pick) for pick in range(start, end + 1)]
    positive = [v for v in values if v > 0.0]
    if not positive:
        return _DRAFT_ROUND_FLOOR_COST
    return max(sum(positive) / len(positive), _DRAFT_ROUND_FLOOR_COST)


def picks_to_dollar_costs(
    entries: list[tuple[int, int]],
    season: int,
    league_name: str,
    league: LeagueSettings,
    curve: PickValueCurve,
) -> list[KeeperCost]:
    """Convert (player_id, round) pairs to dollar-denominated KeeperCost records."""
    return [
        KeeperCost(
            player_id=player_id,
            season=season,
            league=league_name,
            cost=round_to_dollar_cost(round_num, league, curve),
            source="draft_round",
        )
        for player_id, round_num in entries
    ]


def compute_pick_value_curve(
    adp: list[ADP],
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str] | None = None,
) -> PickValueCurve:
    """Build a smooth, monotonically non-increasing pick value curve.

    Joins ADP entries to valuations by player_id, then interpolates gaps,
    applies rolling-average smoothing, and enforces monotonicity.
    """
    total_picks = league.teams * (league.roster_batters + league.roster_pitchers)

    # Extract metadata from the first ADP/valuation entry
    season = adp[0].season if adp else 0
    provider = adp[0].provider if adp else ""
    system = valuations[0].system if valuations else ""

    # Build valuation lookup (use max value if multiple per player)
    val_by_id: dict[int, float] = {}
    for v in valuations:
        if v.player_id not in val_by_id or v.value > val_by_id[v.player_id]:
            val_by_id[v.player_id] = v.value

    # Build raw pick→(values, names) mapping from ADP-valuation join
    raw_values: dict[int, list[float]] = {}
    raw_names: dict[int, list[str]] = {}
    for entry in adp:
        if entry.player_id not in val_by_id:
            continue
        pick_num = round(entry.overall_pick)
        if pick_num < 1:
            continue
        raw_values.setdefault(pick_num, []).append(val_by_id[entry.player_id])
        if player_names and entry.player_id in player_names:
            raw_names.setdefault(pick_num, []).append(player_names[entry.player_id])

    # Average multiple players at the same pick
    averaged: dict[int, float] = {}
    for pick_num, values in raw_values.items():
        averaged[pick_num] = sum(values) / len(values)

    # Track which picks have direct data vs interpolated vs extrapolated
    direct_picks: set[int] = set(averaged.keys())

    # Interpolate gaps between known points
    filled = _interpolate(averaged, total_picks)

    # Track interpolated picks (between known points, not direct)
    interpolated_picks: set[int] = set()
    if direct_picks:
        min_direct = min(direct_picks)
        max_direct = max(direct_picks)
        for pick_num in range(min_direct, min(max_direct + 1, total_picks + 1)):
            if pick_num not in direct_picks:
                interpolated_picks.add(pick_num)

    # Apply rolling-average smoothing
    smoothed = _smooth(filled, total_picks, _SMOOTHING_WINDOW)

    # Enforce monotonicity: walk from pick 1 to N
    mono = _enforce_monotonicity(smoothed, total_picks)

    # Build PickValue list
    picks: list[PickValue] = []
    for pick_num in range(1, total_picks + 1):
        value = mono.get(pick_num, 0.0)

        # Determine confidence
        if pick_num in direct_picks:
            confidence = "high"
        elif pick_num in interpolated_picks:
            confidence = "medium"
        else:
            confidence = "low"

        # Determine player name (use first name from direct data if available)
        name: str | None = None
        if pick_num in raw_names:
            name = raw_names[pick_num][0]

        picks.append(
            PickValue(
                pick=pick_num,
                expected_value=round(value, 2),
                player_name=name,
                confidence=confidence,
            )
        )

    return PickValueCurve(
        season=season,
        provider=provider,
        system=system,
        picks=picks,
        total_picks=total_picks,
    )


def _interpolate(known: dict[int, float], total_picks: int) -> dict[int, float]:
    """Fill gaps between known pick values with linear interpolation."""
    if not known:
        return {i: 0.0 for i in range(1, total_picks + 1)}

    result = dict(known)
    sorted_picks = sorted(known.keys())

    # Fill before the first known pick (extrapolate flat)
    if sorted_picks[0] > 1:
        for pick_num in range(1, sorted_picks[0]):
            result[pick_num] = known[sorted_picks[0]]

    # Interpolate between known points
    for i in range(len(sorted_picks) - 1):
        p1, p2 = sorted_picks[i], sorted_picks[i + 1]
        v1, v2 = known[p1], known[p2]
        for pick_num in range(p1 + 1, p2):
            fraction = (pick_num - p1) / (p2 - p1)
            result[pick_num] = v1 + fraction * (v2 - v1)

    # Extrapolate after the last known pick (linear decay toward 0)
    if sorted_picks[-1] < total_picks:
        last_pick = sorted_picks[-1]
        last_val = known[last_pick]
        remaining = total_picks - last_pick
        for pick_num in range(last_pick + 1, total_picks + 1):
            fraction = (pick_num - last_pick) / (remaining + 1)
            result[pick_num] = max(0.0, last_val * (1 - fraction))

    return result


def _smooth(values: dict[int, float], total_picks: int, window: int) -> dict[int, float]:
    """Apply rolling-average smoothing."""
    result: dict[int, float] = {}
    half = window // 2
    for pick_num in range(1, total_picks + 1):
        start = max(1, pick_num - half)
        end = min(total_picks, pick_num + half)
        window_vals = [values[i] for i in range(start, end + 1) if i in values]
        result[pick_num] = sum(window_vals) / len(window_vals) if window_vals else 0.0
    return result


def _enforce_monotonicity(values: dict[int, float], total_picks: int) -> dict[int, float]:
    """Ensure values are monotonically non-increasing from pick 1 to N."""
    result: dict[int, float] = {}
    prev = float("inf")
    for pick_num in range(1, total_picks + 1):
        val = values.get(pick_num, 0.0)
        val = min(val, prev)
        result[pick_num] = val
        prev = val
    return result


def _pick_value_from_curve(curve: PickValueCurve, pick: int) -> PickValue:
    """Look up the PickValue entry for a pick number from the curve."""
    for pv in curve.picks:
        if pv.pick == pick:
            return pv
    return PickValue(pick=pick, expected_value=0.0, player_name=None, confidence="low")


def evaluate_pick_trade(
    trade: PickTrade,
    curve: PickValueCurve,
    threshold: float = 1.0,
) -> PickTradeEvaluation:
    """Evaluate a draft pick trade by comparing total expected value on each side."""
    gives_detail = [_pick_value_from_curve(curve, p) for p in trade.gives]
    receives_detail = [_pick_value_from_curve(curve, p) for p in trade.receives]

    gives_value = sum(pv.expected_value for pv in gives_detail)
    receives_value = sum(pv.expected_value for pv in receives_detail)
    net_value = receives_value - gives_value

    if net_value >= threshold:
        recommendation = "accept"
    elif net_value <= -threshold:
        recommendation = "reject"
    else:
        recommendation = "even"

    return PickTradeEvaluation(
        trade=trade,
        gives_value=gives_value,
        receives_value=receives_value,
        net_value=net_value,
        gives_detail=gives_detail,
        receives_detail=receives_detail,
        recommendation=recommendation,
    )


def _snake_order(teams: int, total_picks: int) -> dict[int, int]:
    """Build a pick_number→team_idx mapping for snake draft order."""
    order: dict[int, int] = {}
    for pick_num in range(1, total_picks + 1):
        zero_based = pick_num - 1
        round_number = zero_based // teams
        position_in_round = zero_based % teams
        if round_number % 2 == 0:
            order[pick_num] = position_in_round
        else:
            order[pick_num] = teams - 1 - position_in_round
    return order


def _apply_trade(
    pick_order: dict[int, int],
    trade: PickTrade,
    user_team_idx: int,
) -> dict[int, int]:
    """Return a new pick_order with the trade applied.

    Swaps ownership of gives/receives picks between user and trade partner.
    Raises ValueError if picks don't belong to expected teams.
    """
    # Validate gives picks belong to user
    for pick in trade.gives:
        if pick_order[pick] != user_team_idx:
            msg = f"Pick {pick} does not belong to user (team {user_team_idx})"
            raise ValueError(msg)

    # Validate receives picks don't belong to user
    partner_teams: set[int] = set()
    for pick in trade.receives:
        if pick_order[pick] == user_team_idx:
            msg = f"Pick {pick} already belongs to user (team {user_team_idx})"
            raise ValueError(msg)
        partner_teams.add(pick_order[pick])

    # Validate all receives come from the same partner
    if len(partner_teams) > 1:
        msg = f"Receives picks come from multiple teams: {partner_teams}"
        raise ValueError(msg)

    partner_team = next(iter(partner_teams))
    new_order = dict(pick_order)
    for pick in trade.gives:
        new_order[pick] = partner_team
    for pick in trade.receives:
        new_order[pick] = user_team_idx
    return new_order


# Composite slot eligibility for cascade draft
_CASCADE_COMPOSITE_SLOTS: dict[str, list[str]] = {
    "2B": ["MI"],
    "SS": ["MI"],
    "1B": ["CI"],
    "3B": ["CI"],
}

_PITCHER_TYPES = {"P", "pitcher"}


def _cascade_compute_needs(roster: list[DraftPick], slots: dict[str, int]) -> dict[str, int]:
    """Return unfilled slot counts for a team."""
    filled: dict[str, int] = {}
    for pick in roster:
        filled[pick.position] = filled.get(pick.position, 0) + 1
    return {pos: total - filled.get(pos, 0) for pos, total in slots.items() if total - filled.get(pos, 0) > 0}


def _cascade_find_slot(player: DraftBoardRow, needs: dict[str, int]) -> str | None:
    """Find a roster slot for a player given current needs.

    Priority: primary position > composite (MI/CI) > flex (UTIL/P).
    """
    pos = player.position
    if pos in needs and needs[pos] > 0:
        return pos
    if player.player_type not in _PITCHER_TYPES:
        for composite in _CASCADE_COMPOSITE_SLOTS.get(pos, []):
            if composite in needs and needs[composite] > 0:
                return composite
        if "UTIL" in needs and needs["UTIL"] > 0:
            return "UTIL"
    else:
        if "P" in needs and needs["P"] > 0:
            return "P"
    return None


def _run_greedy_draft(
    board: DraftBoard,
    league: LeagueSettings,
    pick_order: dict[int, int],
) -> dict[int, list[DraftPick]]:
    """Simulate a greedy draft where each team picks the highest-value fitting player."""
    slots = build_draft_roster_slots(league)
    num_teams = len(set(pick_order.values()))
    total_picks = len(pick_order)

    pool = sorted(board.rows, key=lambda r: r.value, reverse=True)
    pool_ids: set[int] = {r.player_id for r in pool}
    rosters: dict[int, list[DraftPick]] = {i: [] for i in range(num_teams)}

    for pick_num in range(1, total_picks + 1):
        team_idx = pick_order[pick_num]
        needs = _cascade_compute_needs(rosters[team_idx], slots)

        # Find highest-value assignable player
        chosen_row: DraftBoardRow | None = None
        chosen_slot: str | None = None
        for row in pool:
            if row.player_id not in pool_ids:
                continue
            slot = _cascade_find_slot(row, needs)
            if slot is not None:
                chosen_row = row
                chosen_slot = slot
                break

        if chosen_row is None or chosen_slot is None:
            continue  # no assignable player left

        round_num = (pick_num - 1) // num_teams + 1
        draft_pick = DraftPick(
            round=round_num,
            pick=pick_num,
            team_idx=team_idx,
            player_id=chosen_row.player_id,
            player_name=chosen_row.player_name,
            position=chosen_slot,
            value=chosen_row.value,
        )
        rosters[team_idx].append(draft_pick)
        pool_ids.discard(chosen_row.player_id)
        pool = [r for r in pool if r.player_id != chosen_row.player_id]

    return rosters


def _best_player_at_pick(
    pick: int,
    board: DraftBoard,
    needed_positions: list[str],
    window: int = 2,
) -> PickValue | None:
    """Find the best player matching a needed position near a pick number.

    Searches board rows whose rank is within [pick - window, pick + window].
    Returns a PickValue if a matching player is found, otherwise None.
    """
    candidates = [row for row in board.rows if abs(row.rank - pick) <= window and row.position in needed_positions]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r.value)
    return PickValue(
        pick=pick,
        expected_value=best.value,
        player_name=best.player_name,
        confidence="high",
    )


def evaluate_pick_trade_with_context(
    trade: PickTrade,
    curve: PickValueCurve,
    board: DraftBoard,
    needed_positions: list[str],
    threshold: float = 1.0,
) -> PickTradeEvaluation:
    """Evaluate a draft pick trade with positional-need context.

    For each pick, looks for a player at a needed position near that pick.
    If found, uses that player's actual value; otherwise falls back to curve.
    """
    if not needed_positions:
        return evaluate_pick_trade(trade, curve, threshold)

    gives_detail: list[PickValue] = []
    for p in trade.gives:
        player_pv = _best_player_at_pick(p, board, needed_positions)
        gives_detail.append(player_pv or _pick_value_from_curve(curve, p))

    receives_detail: list[PickValue] = []
    for p in trade.receives:
        player_pv = _best_player_at_pick(p, board, needed_positions)
        receives_detail.append(player_pv or _pick_value_from_curve(curve, p))

    gives_value = sum(pv.expected_value for pv in gives_detail)
    receives_value = sum(pv.expected_value for pv in receives_detail)
    net_value = receives_value - gives_value

    if net_value >= threshold:
        recommendation = "accept"
    elif net_value <= -threshold:
        recommendation = "reject"
    else:
        recommendation = "even"

    return PickTradeEvaluation(
        trade=trade,
        gives_value=gives_value,
        receives_value=receives_value,
        net_value=net_value,
        gives_detail=gives_detail,
        receives_detail=receives_detail,
        recommendation=recommendation,
    )


def cascade_analysis(
    trade: PickTrade,
    board: DraftBoard,
    league: LeagueSettings,
    user_team_idx: int,
    threshold: float = 1.0,
) -> CascadeResult:
    """Analyze how a pick trade affects the entire draft via greedy simulation.

    Runs two full greedy drafts — one with the original pick order, one with the
    traded pick order — and compares the user's roster quality across scenarios.
    """
    total_picks = league.teams * (league.roster_batters + league.roster_pitchers)
    order_before = _snake_order(league.teams, total_picks)
    order_after = _apply_trade(order_before, trade, user_team_idx)

    rosters_before = _run_greedy_draft(board, league, order_before)
    rosters_after = _run_greedy_draft(board, league, order_after)

    before_picks = rosters_before[user_team_idx]
    after_picks = rosters_after[user_team_idx]
    before_value = sum(p.value for p in before_picks)
    after_value = sum(p.value for p in after_picks)
    value_delta = after_value - before_value

    if value_delta >= threshold:
        recommendation = "accept"
    elif value_delta <= -threshold:
        recommendation = "reject"
    else:
        recommendation = "even"

    return CascadeResult(
        trade=trade,
        before=CascadeRoster(picks=before_picks, total_value=before_value),
        after=CascadeRoster(picks=after_picks, total_value=after_value),
        value_delta=value_delta,
        recommendation=recommendation,
    )
