from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import YahooLeague


def _renew_to_league_key(renew: str) -> str:
    """Convert Yahoo renew format ``"game_key_league_id"`` to ``"game_key.l.league_id"``."""
    return renew.replace("_", ".l.", 1)


def find_league_lineage(leagues: list[YahooLeague], start_key: str) -> list[str]:
    """Walk the renewal chain from *start_key* using in-memory league data.

    Returns league keys ordered by season (oldest first).
    """
    by_key = {lg.league_key: lg for lg in leagues}

    chain: list[tuple[int, str]] = []
    current_key = start_key

    visited: set[str] = set()
    while current_key in by_key and current_key not in visited:
        visited.add(current_key)
        league = by_key[current_key]
        chain.append((league.season, league.league_key))
        if not league.renew:
            break
        current_key = _renew_to_league_key(league.renew)

    chain.sort(key=lambda x: x[0])
    return [key for _, key in chain]
