from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.il_stint import ILStint


@dataclass(frozen=True)
class InjuryProfile:
    player_id: int
    seasons_tracked: int
    total_stints: int
    total_days_lost: int
    avg_days_per_season: float
    max_days_in_season: int
    pct_seasons_with_il: float
    injury_locations: dict[str, int] = field(default_factory=dict)
    recent_stints: list[ILStint] = field(default_factory=list)
    all_stints: list[ILStint] = field(default_factory=list)
