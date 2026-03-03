from dataclasses import dataclass


@dataclass(frozen=True)
class PositionScarcity:
    position: str
    tier_1_value: float
    replacement_value: float
    total_surplus: float
    dropoff_slope: float
    steep_rank: int | None
