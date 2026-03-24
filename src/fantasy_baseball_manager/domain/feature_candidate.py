from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True, slots=True)
class FeatureCandidate:
    name: str
    expression: str
    player_type: PlayerType
    min_pa: int | None
    min_ip: float | None
    created_at: str


@dataclass(frozen=True, slots=True)
class CandidateValue:
    player_id: int
    season: int
    value: float | None


@dataclass(frozen=True, slots=True)
class BinnedValue:
    player_id: int
    season: int
    bin_label: str
    value: float


@dataclass(frozen=True, slots=True)
class BinTargetMean:
    bin_label: str
    target: str
    mean: float
    count: int
