from dataclasses import dataclass

from fantasy_baseball_manager.valuation.models import CategoryValue


@dataclass(frozen=True)
class RosterSlot:
    position: str
    count: int


@dataclass(frozen=True)
class RosterConfig:
    slots: tuple[RosterSlot, ...]


@dataclass(frozen=True)
class DraftPick:
    player_id: str
    name: str
    is_user: bool
    position: str | None


@dataclass(frozen=True)
class DraftRanking:
    rank: int
    player_id: str
    name: str
    eligible_positions: tuple[str, ...]
    best_position: str | None
    position_multiplier: float
    raw_value: float
    weighted_value: float
    adjusted_value: float
    category_values: tuple[CategoryValue, ...]
