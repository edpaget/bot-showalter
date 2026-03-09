from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from fantasy_baseball_manager.repos import (
        YahooLeagueRepo,
        YahooLeagueSourceProto,
        YahooTeamRepo,
    )

logger = logging.getLogger(__name__)


def _renew_to_league_key(renew: str) -> str:
    """Convert Yahoo renew format ``"game_key_league_id"`` to ``"game_key.l.league_id"``."""
    return renew.replace("_", ".l.", 1)


def walk_renewal_chain(
    league_source: YahooLeagueSourceProto,
    league_repo: YahooLeagueRepo,
    team_repo: YahooTeamRepo,
    league_key: str,
    *,
    since: int | None = None,
) -> list[tuple[str, int]]:
    """Walk the Yahoo league renewal chain and return all ``(league_key, season)`` pairs.

    Starts from *league_key*, syncs metadata if not already in the DB, reads the
    ``renew`` field to discover the prior-season league, and repeats until the
    chain ends or *since* is reached.

    Returns pairs newest-first.  Logs a warning and returns a partial list when
    the chain breaks (``renew`` is ``None``/empty or the API returns an error).
    """
    result: list[tuple[str, int]] = []
    current_key = league_key

    while True:
        # Sync metadata if not already in DB
        league = league_repo.get_by_league_key(current_key)
        if league is None:
            game_key = current_key.split(".l.")[0]
            try:
                league, teams = league_source.fetch(
                    league_key=current_key,
                    game_key=game_key,
                )
            except httpx.HTTPStatusError:
                logger.warning(
                    "Failed to fetch league metadata for %s — stopping chain walk",
                    current_key,
                )
                break
            league_repo.upsert(league)
            for team in teams:
                team_repo.upsert(team)

        result.append((league.league_key, league.season))

        # Check if we should stop
        if since is not None and league.season <= since:
            break

        renew = league.renew
        if not renew:
            break

        current_key = _renew_to_league_key(renew)

    return result
