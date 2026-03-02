"""Mock draft simulation engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.domain.mock_draft import DraftPick, DraftResult
from fantasy_baseball_manager.services.draft_state import build_draft_roster_slots

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.domain.draft_board import DraftBoard, DraftBoardRow
    from fantasy_baseball_manager.domain.league_settings import LeagueSettings


class DraftBot(Protocol):
    """Protocol for mock draft bot strategies."""

    def pick(
        self,
        available: list[DraftBoardRow],
        roster: list[DraftPick],
        league: LeagueSettings,
    ) -> int:
        """Return the player_id to draft from the available list."""
        ...


def _snake_team(pick_number: int, teams: int) -> int:
    """Return 0-indexed team for a 1-indexed pick number in snake order."""
    zero_based = pick_number - 1
    round_number = zero_based // teams
    position_in_round = zero_based % teams
    if round_number % 2 == 0:
        return position_in_round
    return teams - 1 - position_in_round


def _compute_needs(roster: list[DraftPick], slots: dict[str, int]) -> dict[str, int]:
    """Return unfilled slot counts for a team."""
    filled: dict[str, int] = {}
    for pick in roster:
        filled[pick.position] = filled.get(pick.position, 0) + 1
    needs: dict[str, int] = {}
    for pos, total in slots.items():
        remaining = total - filled.get(pos, 0)
        if remaining > 0:
            needs[pos] = remaining
    return needs


# Composite slot eligibility: position -> list of composite slots it can fill
_COMPOSITE_SLOTS: dict[str, list[str]] = {
    "2B": ["MI"],
    "SS": ["MI"],
    "1B": ["CI"],
    "3B": ["CI"],
}


def _assign_position(player: DraftBoardRow, needs: dict[str, int]) -> str | None:
    """Assign a roster slot for the player based on needs.

    Returns the position string if assignable, None otherwise.
    Priority: primary position > composite slots (MI/CI) > flex (UTIL/P).
    """
    pos = player.position

    # Check primary position
    if pos in needs and needs[pos] > 0:
        return pos

    # Check composite slots (MI, CI)
    if player.player_type == "B":
        for composite in _COMPOSITE_SLOTS.get(pos, []):
            if composite in needs and needs[composite] > 0:
                return composite
        # Flex: UTIL for batters
        if "UTIL" in needs and needs["UTIL"] > 0:
            return "UTIL"
    elif player.player_type == "P":
        if "P" in needs and needs["P"] > 0:
            return "P"

    return None


def _available_for_team(
    pool: list[DraftBoardRow],
    needs: dict[str, int],
) -> list[DraftBoardRow]:
    """Filter pool to players assignable to this team, sorted by value descending."""
    assignable = [p for p in pool if _assign_position(p, needs) is not None]
    return sorted(assignable, key=lambda p: p.value, reverse=True)


def run_mock_draft(
    board: DraftBoard,
    league: LeagueSettings,
    strategies: Sequence[DraftBot],
    *,
    snake: bool = True,
    seed: int | None = None,
) -> DraftResult:
    """Run a complete mock draft simulation.

    Args:
        board: The draft board with ranked players.
        league: League settings defining roster slots.
        strategies: One bot per team (len must equal number of teams).
        snake: Use snake draft ordering (default True).
        seed: Random seed for reproducibility (reserved for future use;
              bots should be seeded externally).

    Returns:
        DraftResult with all picks and rosters.
    """
    _ = seed  # reserved for future use

    slots = build_draft_roster_slots(league)
    num_teams = len(strategies)
    total_rounds = sum(slots.values())
    total_picks = num_teams * total_rounds

    # Build mutable pool sorted by value descending
    pool = sorted(board.rows, key=lambda r: r.value, reverse=True)
    pool_set = {r.player_id for r in pool}

    rosters: dict[int, list[DraftPick]] = {i: [] for i in range(num_teams)}
    picks: list[DraftPick] = []

    for pick_num in range(1, total_picks + 1):
        team_idx = _snake_team(pick_num, num_teams) if snake else (pick_num - 1) % num_teams

        round_num = (pick_num - 1) // num_teams + 1
        needs = _compute_needs(rosters[team_idx], slots)
        available = _available_for_team(pool, needs)

        if not available:
            msg = f"No assignable players for team {team_idx} at pick {pick_num} (round {round_num}). Needs: {needs}"
            raise RuntimeError(msg)

        # Let the bot choose
        bot = strategies[team_idx]
        player_id = bot.pick(available, rosters[team_idx], league)

        # Find the chosen player in pool
        chosen: DraftBoardRow | None = None
        for row in pool:
            if row.player_id == player_id:
                chosen = row
                break

        if chosen is None or player_id not in pool_set:
            # Bot picked an invalid player — fall back to first available
            chosen = available[0]

        position = _assign_position(chosen, needs)
        if position is None:
            # Shouldn't happen since we filtered, but safety fallback
            chosen = available[0]
            position = _assign_position(chosen, needs)
            if position is None:  # pragma: no cover
                msg = f"Cannot assign position for {chosen.player_name}"
                raise RuntimeError(msg)

        draft_pick = DraftPick(
            round=round_num,
            pick=pick_num,
            team_idx=team_idx,
            player_id=chosen.player_id,
            player_name=chosen.player_name,
            position=position,
            value=chosen.value,
        )

        picks.append(draft_pick)
        rosters[team_idx].append(draft_pick)

        # Remove from pool
        pool = [r for r in pool if r.player_id != chosen.player_id]
        pool_set.discard(chosen.player_id)

    return DraftResult(picks=picks, rosters=rosters, snake=snake)
