from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class TeamCategoryProjection:
    category: str
    projected_value: float
    league_rank_estimate: int
    strength: str


@dataclass(frozen=True)
class RosterAnalysis:
    projections: list[TeamCategoryProjection]
    strongest_categories: list[str]
    weakest_categories: list[str]


@dataclass(frozen=True)
class PlayerRecommendation:
    player_id: int
    player_name: str
    player_type: PlayerType
    category_impact: float
    tradeoff_categories: tuple[str, ...]


@dataclass(frozen=True)
class CategoryNeed:
    category: str
    current_rank: int
    target_rank: int
    best_available: tuple[PlayerRecommendation, ...]
