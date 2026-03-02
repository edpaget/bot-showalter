from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.keeper import KeeperDecision


@dataclass(frozen=True)
class KeeperConstraints:
    max_keepers: int
    max_per_position: dict[str, int] | None = None
    max_cost: float | None = None
    required_keepers: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class KeeperSet:
    players: tuple[KeeperDecision, ...]
    total_surplus: float
    total_cost: float
    positions_filled: dict[str, int]
    score: float


@dataclass(frozen=True)
class SensitivityEntry:
    player_name: str
    player_id: int
    surplus_gap: float


@dataclass(frozen=True)
class KeeperSolution:
    optimal: KeeperSet
    alternatives: list[KeeperSet]
    sensitivity: list[SensitivityEntry]
