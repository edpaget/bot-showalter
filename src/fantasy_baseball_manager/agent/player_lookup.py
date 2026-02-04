"""Player name matching utilities for agent tools.

This module provides shared functions for finding players by name,
eliminating duplication across lookup_player, compare_players, rank_keepers,
and get_player_info tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.valuation.models import PlayerValue


def find_player_by_name(
    name: str,
    all_values: list[PlayerValue],
) -> PlayerValue | None:
    """Find a single player by name using case-insensitive partial matching.

    Returns the best match (highest total value) if multiple players match,
    or None if no players match.
    """
    name_lower = name.lower()
    matches = [pv for pv in all_values if name_lower in pv.name.lower()]

    if not matches:
        return None

    return max(matches, key=lambda p: p.total_value)


def find_players_by_names(
    names: list[str],
    all_values: list[PlayerValue],
) -> tuple[list[PlayerValue], list[str]]:
    """Find multiple players by name.

    Returns a tuple of (found_players, not_found_names).
    For each name, returns the best match (highest total value) if multiple players match.
    """
    found_players: list[PlayerValue] = []
    not_found: list[str] = []

    for name in names:
        player = find_player_by_name(name, all_values)
        if player:
            found_players.append(player)
        else:
            not_found.append(name)

    return found_players, not_found


def find_all_matches(
    name: str,
    all_values: list[PlayerValue],
) -> list[PlayerValue]:
    """Find all players matching a name using case-insensitive partial matching.

    Returns all matches sorted by total value (highest first).
    """
    name_lower = name.lower()
    matches = [pv for pv in all_values if name_lower in pv.name.lower()]
    return sorted(matches, key=lambda p: p.total_value, reverse=True)
