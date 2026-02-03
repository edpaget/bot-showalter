from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.valuation.models import CategoryValue, PlayerValue


@dataclass(frozen=True)
class SlotCost:
    slot_number: int
    replacement_value: float


@dataclass(frozen=True)
class KeeperCandidate:
    player_id: str
    name: str
    player_value: PlayerValue
    eligible_positions: tuple[str, ...]


@dataclass(frozen=True)
class KeeperSurplus:
    player_id: str
    name: str
    player_value: float
    eligible_positions: tuple[str, ...]
    assigned_slot: int
    replacement_value: float
    surplus_value: float
    category_values: tuple[CategoryValue, ...]


@dataclass(frozen=True)
class KeeperRecommendation:
    keepers: tuple[KeeperSurplus, ...]
    total_surplus: float
    all_candidates: tuple[KeeperSurplus, ...]
