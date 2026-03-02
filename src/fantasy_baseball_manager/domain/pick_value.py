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


def value_at(curve: PickValueCurve, pick: int) -> float:
    """Return expected value for a pick number, 0.0 if out of range."""
    for pv in curve.picks:
        if pv.pick == pick:
            return pv.expected_value
    return 0.0
