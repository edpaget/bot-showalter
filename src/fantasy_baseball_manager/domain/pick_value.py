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
