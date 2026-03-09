from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.services.replacement_padding import blend_projections as _blend
from fantasy_baseball_manager.services.replacement_profiler import compute_replacement_profiles as _compute

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import CategoryConfig, Projection, ReplacementProfile


class ReplacementPaddingService:
    """Concrete implementation of the ReplacementPadder protocol."""

    def compute_replacement_profiles(
        self,
        stats_list: list[dict[str, float]],
        position_map: list[list[str]],
        roster_spots: dict[str, int],
        num_teams: int,
        categories: list[CategoryConfig],
        player_type: str,
    ) -> dict[str, ReplacementProfile]:
        return _compute(stats_list, position_map, roster_spots, num_teams, categories, player_type)

    def blend_projections(
        self,
        projections: list[Projection],
        replacement_profiles: dict[str, ReplacementProfile],
        injury_map: dict[int, float],
        position_map: dict[int, list[str]],
    ) -> list[Projection]:
        return _blend(projections, replacement_profiles, injury_map, position_map)
