import logging
from typing import TYPE_CHECKING, Protocol

from fantasy_baseball_manager.services.draft_state import DraftError

if TYPE_CHECKING:
    from collections.abc import Set

    from fantasy_baseball_manager.domain import YahooDraftPick, YahooTeam
    from fantasy_baseball_manager.services.draft_state import DraftPick

logger = logging.getLogger(__name__)


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


def ingest_yahoo_pick(
    pick_fn: PickFn,
    available_ids: Set[int],
    yahoo_pick: YahooDraftPick,
    team_map: dict[str, int],
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

    try:
        return pick_fn(
            yahoo_pick.player_id,
            team,
            yahoo_pick.position,
            price=yahoo_pick.cost,
        )
    except DraftError as exc:
        logger.warning(
            "Could not ingest Yahoo pick %s: %s",
            yahoo_pick.player_name,
            exc,
        )
        return None
