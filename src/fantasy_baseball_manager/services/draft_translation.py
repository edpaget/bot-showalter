import logging
from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.services.draft_state import DraftError

if TYPE_CHECKING:
    from collections.abc import Set

    from fantasy_baseball_manager.domain import YahooDraftPick, YahooTeam
    from fantasy_baseball_manager.services.draft_state import DraftPick

logger = logging.getLogger(__name__)


_PITCHER_POSITIONS = frozenset({"SP", "RP"})


class PickFn(Protocol):
    def __call__(
        self,
        player_id: int,
        team: int,
        position: str,
        *,
        price: int | None = None,
    ) -> DraftPick: ...


def build_team_map(teams: list[YahooTeam]) -> dict[str, int]:
    return {team.team_key: team.team_id for team in teams}


def resolve_draft_position(
    position: str,
    roster_slots: dict[str, int],
    team_fills: dict[str, int],
) -> str:
    """Map a Yahoo position to the best available roster slot.

    Fallback chain:
    - SP/RP not in roster_slots → try P → try BN
    - Any position with full slot → try UTIL (batters) or P (pitchers) → try BN
    - Returns original position if no fallback applies.
    """

    def _has_room(slot: str) -> bool:
        return slot in roster_slots and team_fills.get(slot, 0) < roster_slots[slot]

    # If the position is a direct roster slot with room, use it
    if _has_room(position):
        return position

    # SP/RP → P fallback
    if position in _PITCHER_POSITIONS:
        if _has_room("P"):
            return "P"
        if _has_room("BN"):
            return "BN"
        return position

    # Batter overflow → UTIL → BN
    if _has_room("UTIL"):
        return "UTIL"
    if _has_room("BN"):
        return "BN"

    return position


def ingest_yahoo_pick(
    pick_fn: PickFn,
    available_ids: Set[int],
    yahoo_pick: YahooDraftPick,
    team_map: dict[str, int],
    *,
    roster_slots: dict[str, int] | None = None,
    team_rosters: dict[int, list[DraftPick]] | None = None,
) -> DraftPick | None:
    if yahoo_pick.player_id is None:
        logger.warning(
            "Skipping unmapped Yahoo pick: %s (%s)",
            yahoo_pick.player_name,
            yahoo_pick.yahoo_player_key,
        )
        return None

    team = team_map.get(yahoo_pick.team_key)
    if team is None:
        logger.warning(
            "Unknown team key %s for pick %s",
            yahoo_pick.team_key,
            yahoo_pick.player_name,
        )
        return None

    if yahoo_pick.player_id not in available_ids:
        logger.warning(
            "Player %s (id=%d) not in available pool — skipping",
            yahoo_pick.player_name,
            yahoo_pick.player_id,
        )
        return None

    position = yahoo_pick.position
    if roster_slots is not None:
        team_picks = (team_rosters or {}).get(team, [])
        team_fills: dict[str, int] = {}
        for p in team_picks:
            team_fills[p.position] = team_fills.get(p.position, 0) + 1
        position = resolve_draft_position(position, roster_slots, team_fills)

    try:
        return pick_fn(
            yahoo_pick.player_id,
            team,
            position,
            price=yahoo_pick.cost,
        )
    except DraftError as exc:
        logger.warning(
            "Could not ingest Yahoo pick %s: %s",
            yahoo_pick.player_name,
            exc,
        )
        return None
