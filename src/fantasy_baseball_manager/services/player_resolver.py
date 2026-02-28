import difflib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import DraftBoardRow


def resolve_player(query: str, pool: list[DraftBoardRow]) -> list[DraftBoardRow]:
    """Resolve a player query against the available pool.

    Resolution order:
    1. Exact full name match (case-insensitive) — single result
    2. Substring match (case-insensitive) — all matches
    3. Fuzzy match via difflib (cutoff=0.6) — all matches
    """
    if not pool:
        return []

    query_lower = query.lower()
    names = {p.player_id: p.player_name.lower() for p in pool}

    # 1. Exact match
    for player in pool:
        if player.player_name.lower() == query_lower:
            return [player]

    # 2. Substring match
    substring_matches = [p for p in pool if query_lower in names[p.player_id]]
    if substring_matches:
        return substring_matches

    # 3. Fuzzy match
    name_list = [p.player_name for p in pool]
    close = difflib.get_close_matches(query, name_list, n=5, cutoff=0.6)
    if close:
        close_set = set(close)
        return [p for p in pool if p.player_name in close_set]

    return []
