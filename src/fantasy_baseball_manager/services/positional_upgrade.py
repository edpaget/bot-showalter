from collections import defaultdict

from fantasy_baseball_manager.domain import (
    DraftBoard,
    DraftBoardRow,
    LeagueSettings,
    MarginalValue,
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
