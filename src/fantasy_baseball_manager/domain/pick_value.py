from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.mock_draft import DraftPick


@dataclass(frozen=True)
class PickValue:
    pick: int
    expected_value: float
    player_name: str | None
    confidence: str  # "high" | "medium" | "low"


@dataclass(frozen=True)
class PickValueCurve:
    season: int
    provider: str
    system: str
    picks: list[PickValue]
    total_picks: int


@dataclass(frozen=True)
class PickTrade:
    gives: list[int]
    receives: list[int]


@dataclass(frozen=True)
class PickTradeEvaluation:
    trade: PickTrade
    gives_value: float
    receives_value: float
    net_value: float
    gives_detail: list[PickValue]
    receives_detail: list[PickValue]
    recommendation: str


@dataclass(frozen=True)
class CascadeRoster:
    picks: list[DraftPick]
    total_value: float


@dataclass(frozen=True)
class CascadeResult:
    trade: PickTrade
    before: CascadeRoster
    after: CascadeRoster
    value_delta: float
    recommendation: str
