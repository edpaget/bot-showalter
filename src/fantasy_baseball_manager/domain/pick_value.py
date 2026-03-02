from dataclasses import dataclass


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
