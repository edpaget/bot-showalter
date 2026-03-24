from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import PlayerType, ReplacementProfile
from fantasy_baseball_manager.models.zar.engine import (
    compute_z_scores,
    convert_rate_stats,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import CategoryConfig


def compute_replacement_profiles(
    stats_list: list[dict[str, float]],
    position_map: list[list[str]],
    roster_spots: dict[str, int],
    num_teams: int,
    categories: list[CategoryConfig],
    player_type: str,
) -> dict[str, ReplacementProfile]:
    """Compute per-position replacement-level stat profiles.

    Returns the original (unconverted) stat line of the replacement-level player
    at each position.  The replacement player is identified the same way as
    ``compute_replacement_level``: sort eligible players by composite z descending,
    pick the one at index ``spots * num_teams``.
    """
    if not stats_list:
        return {}

    category_keys = [c.key for c in categories]
    converted = convert_rate_stats(stats_list, categories)
    z_scores = compute_z_scores(converted, category_keys)

    result: dict[str, ReplacementProfile] = {}
    for position, spots in roster_spots.items():
        eligible = sorted(
            [
                (pz.composite_z, pz.player_index)
                for pz, pos in zip(z_scores, position_map, strict=True)
                if position in pos
            ],
            reverse=True,
        )
        if not eligible:
            stat_line = {c.key: 0.0 for c in categories}
            result[position] = ReplacementProfile(
                position=position,
                player_type=PlayerType(player_type),
                stat_line=stat_line,
            )
            continue

        draftable = spots * num_teams
        repl_index = eligible[-1][1] if draftable >= len(eligible) else eligible[draftable][1]

        result[position] = ReplacementProfile(
            position=position,
            player_type=PlayerType(player_type),
            stat_line=stats_list[repl_index],
        )

    return result
