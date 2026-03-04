from collections import defaultdict

from fantasy_baseball_manager.domain import (
    DraftBoard,
    DraftBoardRow,
    LeagueSettings,
    MarginalValue,
    OpportunityCost,
    PositionUpgrade,
    RosterSlot,
    RosterState,
)

# Positions that are "pitcher" flex slots (generic P)
_PITCHER_POSITIONS = frozenset({"SP", "RP"})


def _is_pitcher_position(pos: str) -> bool:
    return pos in _PITCHER_POSITIONS


def _build_empty_slots(league: LeagueSettings) -> list[RosterSlot]:
    """Expand league position counts into individual RosterSlot objects."""
    slots: list[RosterSlot] = []
    for pos, count in league.positions.items():
        for _ in range(count):
            slots.append(RosterSlot(position=pos))
    for _ in range(league.roster_util):
        slots.append(RosterSlot(position="UTIL"))
    for pos, count in league.pitcher_positions.items():
        for _ in range(count):
            slots.append(RosterSlot(position=pos))
    for _ in range(league.roster_pitchers):
        slots.append(RosterSlot(position="P"))
    return slots


def _slot_scarcity(position: str, league: LeagueSettings) -> int:
    """Return total number of slots for a position (lower = scarcer)."""
    if position == "UTIL":
        return league.roster_util
    if position == "P":
        return league.roster_pitchers
    count = league.positions.get(position, 0)
    if count == 0:
        count = league.pitcher_positions.get(position, 0)
    return count


def build_roster_state(
    drafted_player_ids: list[int],
    board: DraftBoard,
    league: LeagueSettings,
    *,
    position_eligibility: dict[int, list[str]] | None = None,
) -> RosterState:
    """Build a RosterState from drafted players and league settings."""
    slots = _build_empty_slots(league)
    player_lookup: dict[int, DraftBoardRow] = {r.player_id: r for r in board.rows}

    for pid in drafted_player_ids:
        row = player_lookup.get(pid)
        if row is None:
            continue

        eligible = position_eligibility[pid] if position_eligibility and pid in position_eligibility else [row.position]
        # Sort eligible positions by scarcity (fewest slots first)
        eligible_sorted = sorted(eligible, key=lambda p: _slot_scarcity(p, league))

        filled = False
        # Try specific position slots, preferring scarcer positions
        for pos in eligible_sorted:
            for i, slot in enumerate(slots):
                if slot.position == pos and slot.player_id is None:
                    slots[i] = RosterSlot(
                        position=pos,
                        player_id=row.player_id,
                        player_name=row.player_name,
                        value=row.value,
                        category_z_scores=dict(row.category_z_scores),
                    )
                    filled = True
                    break
            if filled:
                break

        if not filled:
            # Try flex slots: UTIL for batters, P for pitchers
            flex_pos = "P" if _is_pitcher_position(row.position) else "UTIL"
            for i, slot in enumerate(slots):
                if slot.position == flex_pos and slot.player_id is None:
                    slots[i] = RosterSlot(
                        position=flex_pos,
                        player_id=row.player_id,
                        player_name=row.player_name,
                        value=row.value,
                        category_z_scores=dict(row.category_z_scores),
                    )
                    break

    open_positions = [s.position for s in slots if s.player_id is None]
    total_value = sum(s.value for s in slots if s.player_id is not None)

    category_totals: dict[str, float] = defaultdict(float)
    for slot in slots:
        if slot.player_id is not None:
            for cat, z in slot.category_z_scores.items():
                category_totals[cat] += z

    return RosterState(
        slots=slots,
        open_positions=open_positions,
        total_value=total_value,
        category_totals=dict(category_totals),
    )


def compute_marginal_values(
    state: RosterState,
    available: list[DraftBoardRow],
    league: LeagueSettings,
    *,
    position_eligibility: dict[int, list[str]] | None = None,
) -> list[MarginalValue]:
    """Compute roster-relative marginal value for each available player."""
    weak_cats = {cat for cat, total in state.category_totals.items() if total < 0}

    # Index slots by position for quick lookup
    slots_by_pos: dict[str, list[RosterSlot]] = defaultdict(list)
    for slot in state.slots:
        slots_by_pos[slot.position].append(slot)

    results: list[MarginalValue] = []

    for row in available:
        eligible = (
            position_eligibility[row.player_id]
            if position_eligibility and row.player_id in position_eligibility
            else [row.position]
        )

        # Compute category-need bonus
        cat_bonus = sum(
            row.category_z_scores.get(cat, 0.0) for cat in weak_cats if row.category_z_scores.get(cat, 0.0) > 0
        )

        best_mv = 0.0
        best_pos = eligible[0] if eligible else row.position
        best_fills_need = False
        best_upgrade_over: str | None = None

        # Also check flex slot
        flex_pos = "P" if _is_pitcher_position(row.position) else "UTIL"
        positions_to_check = list(eligible) + [flex_pos]

        for pos in positions_to_check:
            pos_slots = slots_by_pos.get(pos, [])
            if not pos_slots:
                continue

            # Check for open slot
            has_open = any(s.player_id is None for s in pos_slots)
            if has_open:
                mv = row.value + cat_bonus
                if mv > best_mv:
                    best_mv = mv
                    best_pos = pos
                    best_fills_need = True
                    best_upgrade_over = None
            else:
                # Check for upgrade: find worst player at this position
                worst = min(pos_slots, key=lambda s: s.value)
                if row.value > worst.value:
                    upgrade_diff = row.value - worst.value
                    # Category impact bonus for upgrades
                    cat_impact_bonus = sum(
                        row.category_z_scores.get(cat, 0.0) - worst.category_z_scores.get(cat, 0.0)
                        for cat in weak_cats
                        if (row.category_z_scores.get(cat, 0.0) - worst.category_z_scores.get(cat, 0.0)) > 0
                    )
                    mv = upgrade_diff + cat_impact_bonus
                    if mv > best_mv:
                        best_mv = mv
                        best_pos = pos
                        best_fills_need = False
                        best_upgrade_over = worst.player_name

        category_impacts = dict(row.category_z_scores)

        results.append(
            MarginalValue(
                player_id=row.player_id,
                player_name=row.player_name,
                position=best_pos,
                raw_value=row.value,
                marginal_value=best_mv,
                category_impacts=category_impacts,
                fills_need=best_fills_need,
                upgrade_over=best_upgrade_over,
            )
        )

    results.sort(key=lambda m: m.marginal_value, reverse=True)
    return results


# Flex positions that don't get their own upgrade analysis
_FLEX_POSITIONS = frozenset({"UTIL", "P"})

_URGENCY_ORDER = {"high": 0, "medium": 1, "low": 2}


def compute_position_upgrades(
    state: RosterState,
    available: list[DraftBoardRow],
    league: LeagueSettings,
    *,
    position_eligibility: dict[int, list[str]] | None = None,
    high_dropoff_threshold: float = 5.0,
) -> list[PositionUpgrade]:
    """Compute per-position upgrade comparison for the current roster."""
    # Collect non-flex positions
    positions: list[str] = []
    for pos in league.positions:
        if pos not in _FLEX_POSITIONS:
            positions.append(pos)
    for pos in league.pitcher_positions:
        if pos not in _FLEX_POSITIONS:
            positions.append(pos)

    # Build eligibility index: position → list of available rows sorted by value desc
    pos_available: dict[str, list[DraftBoardRow]] = {pos: [] for pos in positions}
    for row in available:
        eligible = (
            position_eligibility[row.player_id]
            if position_eligibility and row.player_id in position_eligibility
            else [row.position]
        )
        for pos in eligible:
            if pos in pos_available:
                pos_available[pos].append(row)

    for pos in pos_available:
        pos_available[pos].sort(key=lambda r: r.value, reverse=True)

    upgrades: list[PositionUpgrade] = []

    for pos in positions:
        candidates = pos_available[pos]
        if not candidates:
            continue

        best = candidates[0]
        next_best_row = candidates[1] if len(candidates) > 1 else None

        # Find current starter(s) at this position
        pos_slots = [s for s in state.slots if s.position == pos]
        filled_slots = [s for s in pos_slots if s.player_id is not None]
        is_open = len(filled_slots) < len(pos_slots)

        if filled_slots:
            # Use the worst starter as the upgrade target
            worst = min(filled_slots, key=lambda s: s.value)
            current_player = worst.player_name
            current_value = worst.value
        else:
            current_player = None
            current_value = 0.0

        upgrade_value = best.value - current_value
        dropoff = best.value - next_best_row.value if next_best_row else best.value

        urgency = ("high" if dropoff >= high_dropoff_threshold else "medium") if is_open else "low"

        upgrades.append(
            PositionUpgrade(
                position=pos,
                current_player=current_player,
                current_value=current_value,
                best_available=best.player_name,
                best_available_value=best.value,
                upgrade_value=upgrade_value,
                next_best=next_best_row.player_name if next_best_row else None,
                dropoff_to_next=dropoff,
                urgency=urgency,
            )
        )

    upgrades.sort(key=lambda u: (_URGENCY_ORDER[u.urgency], -u.upgrade_value))
    return upgrades


def compute_opportunity_costs(
    marginal_values: list[MarginalValue],
    state: RosterState,
    league: LeagueSettings,
    picks_until_next: int,
) -> list[OpportunityCost]:
    """Score position-fill candidates by opportunity cost of drafting them now."""
    # "Will be gone" set: top picks_until_next players by MV (already sorted desc)
    gone_set = marginal_values[:picks_until_next]

    results: list[OpportunityCost] = []

    for candidate in marginal_values:
        if not candidate.fills_need:
            continue

        # Best non-fill player in the gone set (excluding the candidate itself)
        best_non_fill_mv = 0.0
        for gone in gone_set:
            if gone.player_id == candidate.player_id:
                continue
            if not gone.fills_need and gone.marginal_value > best_non_fill_mv:
                best_non_fill_mv = gone.marginal_value

        net = candidate.marginal_value - best_non_fill_mv

        if net > 0:
            rec = "draft now"
        elif net < 0:
            rec = "wait"
        else:
            rec = "borderline"

        results.append(
            OpportunityCost(
                position=candidate.position,
                recommended_player=candidate.player_name,
                marginal_value=candidate.marginal_value,
                opportunity_cost=best_non_fill_mv,
                net_value=net,
                recommendation=rec,
            )
        )

    results.sort(key=lambda r: r.net_value, reverse=True)
    return results
