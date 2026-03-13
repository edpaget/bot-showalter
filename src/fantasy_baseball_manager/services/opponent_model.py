from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import LeagueNeeds, PositionRun, TeamNeeds, ThreatAssessment
from fantasy_baseball_manager.services.draft_state import DraftEngine, DraftFormat, PoolKey, build_draft_roster_slots

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoardRow, LeagueSettings
    from fantasy_baseball_manager.services.draft_state import DraftState

# Composite slot eligibility for supply counting
_COMPOSITE_POSITIONS: dict[str, list[str]] = {
    "MI": ["2B", "SS"],
    "CI": ["1B", "3B"],
}


def _count_supply(
    pool: dict[PoolKey, DraftBoardRow],
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


def _is_clustered(pick_numbers: list[int], half_round: int) -> bool:
    """Check if any contiguous window of size half_round contains 2+ picks."""
    if len(pick_numbers) < 2:
        return False
    sorted_picks = sorted(pick_numbers)
    for i in range(len(sorted_picks)):
        for j in range(i + 1, len(sorted_picks)):
            if sorted_picks[j] - sorted_picks[i] < half_round:
                return True
    return False


def detect_position_runs(
    state: DraftState,
    league: LeagueSettings,
    *,
    window: int | None = None,
) -> list[PositionRun]:
    """Detect position runs in recent draft picks.

    A "run" is when multiple teams draft the same position in a short span,
    signalling rapidly shrinking supply.
    """
    if window is None:
        window = 2 * state.config.teams

    half_round = state.config.teams // 2
    recent_picks = state.picks[-window:] if state.picks else []

    # Group recent picks by position
    by_position: dict[str, list[int]] = {}
    for pick in recent_picks:
        by_position.setdefault(pick.position, []).append(pick.pick_number)

    # Compute user's remaining needs
    slots = build_draft_roster_slots(league)
    user_roster = state.team_rosters.get(state.config.user_team, [])
    user_filled: dict[str, int] = {}
    for pick in user_roster:
        user_filled[pick.position] = user_filled.get(pick.position, 0) + 1

    runs: list[PositionRun] = []
    for position, pick_numbers in by_position.items():
        if len(pick_numbers) < 2:
            continue

        if not _is_clustered(pick_numbers, half_round):
            continue

        remaining_supply = _count_supply(state.available_pool, position)
        user_need = slots.get(position, 0) - user_filled.get(position, 0)
        user_need = max(0, user_need)

        run_length = len(pick_numbers)

        if run_length >= 3 and user_need > 0 and remaining_supply < 1.5 * user_need:
            urgency = "critical"
        else:
            urgency = "developing"

        runs.append(
            PositionRun(
                position=position,
                pick_numbers=tuple(sorted(pick_numbers)),
                run_length=run_length,
                remaining_supply=remaining_supply,
                urgency=urgency,
            )
        )

    # Sort: critical first, then by remaining_supply ascending
    urgency_order = {"critical": 0, "developing": 1}
    runs.sort(key=lambda r: (urgency_order[r.urgency], r.remaining_supply))

    return runs


def _picks_until_user_next(state: DraftState) -> int:
    """How many picks until the user selects again in a snake draft.

    Returns 0 for auction format (anyone can bid anytime).
    """
    if state.config.format in (DraftFormat.AUCTION, DraftFormat.LIVE):
        return 0

    teams = state.config.teams
    user = state.config.user_team
    current = state.current_pick

    if DraftEngine._snake_team(current, teams) == user:
        return 0

    for offset in range(1, 2 * teams):
        future = current + offset
        if DraftEngine._snake_team(future, teams) == user:
            return offset

    return 2 * teams


def _is_user_need(player: DraftBoardRow, needs: dict[str, int]) -> bool:
    """Check if a player can fill any open roster slot for the user."""
    if player.position in needs:
        return True
    if player.player_type == "batter" and "UTIL" in needs:
        return True
    return player.player_type == "pitcher" and "P" in needs


def _classify_threat(
    adp: float | None,
    current_pick: int,
    picks_until: int,
    teams_needing: int,
) -> str:
    """Classify threat level based on ADP and opponent needs."""
    danger_zone = current_pick + picks_until
    adp_in_danger = adp is not None and adp < danger_zone

    if adp_in_danger and teams_needing >= 2:
        return "likely-gone"
    if (adp_in_danger and teams_needing >= 1) or teams_needing >= 3:
        return "at-risk"
    return "safe"


def assess_threats(
    state: DraftState,
    league: LeagueSettings,
    *,
    limit: int = 10,
) -> list[ThreatAssessment]:
    """Assess which available players are at risk of being taken before the user's next pick."""
    picks_until = _picks_until_user_next(state)

    # Auction: everyone can bid on anyone, no meaningful threat ordering
    if picks_until == 0:
        return []

    # Identify teams picking between current_pick (exclusive) and user's next pick (exclusive)
    teams = state.config.teams
    intervening_teams: list[int] = []
    for offset in range(1, picks_until):
        pick_num = state.current_pick + offset
        team = DraftEngine._snake_team(pick_num, teams)
        if team not in intervening_teams:
            intervening_teams.append(team)

    # Also include the team currently on the clock if it's not the user
    current_team = DraftEngine._snake_team(state.current_pick, teams)
    if current_team != state.config.user_team and current_team not in intervening_teams:
        intervening_teams.insert(0, current_team)

    # Compute unfilled positions for each intervening team
    slots = build_draft_roster_slots(league)
    teams_needing_position: dict[str, int] = {}
    for team_idx in intervening_teams:
        roster = state.team_rosters.get(team_idx, [])
        filled: dict[str, int] = {}
        for pick in roster:
            filled[pick.position] = filled.get(pick.position, 0) + 1
        for slot, total in slots.items():
            if total - filled.get(slot, 0) > 0:
                teams_needing_position[slot] = teams_needing_position.get(slot, 0) + 1

    # User needs for filtering
    user_roster = state.team_rosters.get(state.config.user_team, [])
    user_filled: dict[str, int] = {}
    for pick in user_roster:
        user_filled[pick.position] = user_filled.get(pick.position, 0) + 1
    user_needs: dict[str, int] = {}
    for slot, total in state.config.roster_slots.items():
        remaining = total - user_filled.get(slot, 0)
        if remaining > 0:
            user_needs[slot] = remaining

    # Get top candidates by value
    pool_sorted = sorted(state.available_pool.values(), key=lambda p: p.value, reverse=True)
    candidates = pool_sorted[: limit * 3]

    _THREAT_ORDER = {"likely-gone": 0, "at-risk": 1, "safe": 2}

    assessments: list[ThreatAssessment] = []
    for player in candidates:
        if not _is_user_need(player, user_needs):
            continue

        teams_needing = teams_needing_position.get(player.position, 0)
        threat_level = _classify_threat(
            player.adp_overall,
            state.current_pick,
            picks_until,
            teams_needing,
        )

        assessments.append(
            ThreatAssessment(
                player_id=player.player_id,
                player_name=player.player_name,
                position=player.position,
                value=player.value,
                adp=player.adp_overall,
                picks_until_user_next=picks_until,
                teams_needing_position=teams_needing,
                threat_level=threat_level,
            )
        )

    # Sort: likely-gone first, then at-risk, then safe; within each level by value desc
    assessments.sort(key=lambda a: (_THREAT_ORDER[a.threat_level], -a.value))

    return assessments[:limit]
