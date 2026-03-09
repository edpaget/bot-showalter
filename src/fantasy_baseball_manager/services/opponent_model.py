from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import LeagueNeeds, TeamNeeds
from fantasy_baseball_manager.services.draft_state import build_draft_roster_slots

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoardRow, LeagueSettings
    from fantasy_baseball_manager.services.draft_state import DraftState

# Composite slot eligibility for supply counting
_COMPOSITE_POSITIONS: dict[str, list[str]] = {
    "MI": ["2B", "SS"],
    "CI": ["1B", "3B"],
}


def _count_supply(
    pool: dict[int, DraftBoardRow],
    slot: str,
) -> int:
    """Count available players eligible for a roster slot."""
    players = pool.values()

    if slot in _COMPOSITE_POSITIONS:
        eligible = _COMPOSITE_POSITIONS[slot]
        return sum(1 for p in players if p.position in eligible)

    if slot == "UTIL":
        return sum(1 for p in players if p.player_type == "batter")

    if slot == "P":
        return sum(1 for p in players if p.player_type == "pitcher")

    if slot == "BN":
        return len(pool)

    # Direct position match
    return sum(1 for p in players if p.position == slot)


def compute_league_needs(
    state: DraftState,
    league: LeagueSettings,
    player_values: dict[int, float] | None = None,
) -> LeagueNeeds:
    """Compute per-team unfilled slots and league-wide scarcity ratios."""
    slots = build_draft_roster_slots(league)
    teams: list[TeamNeeds] = []

    for team_idx, roster in state.team_rosters.items():
        filled: dict[str, int] = {}
        for pick in roster:
            filled[pick.position] = filled.get(pick.position, 0) + 1

        unfilled: dict[str, int] = {}
        for slot, total in slots.items():
            remaining = total - filled.get(slot, 0)
            if remaining > 0:
                unfilled[slot] = remaining

        total_value = 0.0
        if player_values:
            total_value = sum(player_values.get(pick.player_id, 0.0) for pick in roster)

        teams.append(
            TeamNeeds(
                team_idx=team_idx,
                team_name=None,
                filled=filled,
                unfilled=unfilled,
                total_value=total_value,
            )
        )

    # Aggregate demand across all teams
    demand_by_position: dict[str, int] = {}
    for team in teams:
        for slot, count in team.unfilled.items():
            demand_by_position[slot] = demand_by_position.get(slot, 0) + count

    # Compute supply from available pool
    all_slots = set(slots.keys())
    supply_by_position: dict[str, int] = {}
    for slot in all_slots:
        supply_by_position[slot] = _count_supply(state.available_pool, slot)

    # Compute scarcity ratio: demand / supply
    scarcity_ratio: dict[str, float] = {}
    for slot in all_slots:
        demand = demand_by_position.get(slot, 0)
        supply = supply_by_position.get(slot, 0)
        if supply == 0:
            scarcity_ratio[slot] = float("inf") if demand > 0 else 0.0
        else:
            scarcity_ratio[slot] = demand / supply

    return LeagueNeeds(
        teams=teams,
        demand_by_position=demand_by_position,
        supply_by_position=supply_by_position,
        scarcity_ratio=scarcity_ratio,
    )
