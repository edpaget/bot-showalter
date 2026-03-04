from dataclasses import dataclass


@dataclass(frozen=True)
class PositionValueCurve:
    position: str
    values: list[tuple[int, str, float]]  # (rank, player_name, value)
    cliff_rank: int | None  # elbow point from _detect_elbow


@dataclass(frozen=True)
class PositionScarcity:
    position: str
    tier_1_value: float
    replacement_value: float
    total_surplus: float
    dropoff_slope: float
    steep_rank: int | None
