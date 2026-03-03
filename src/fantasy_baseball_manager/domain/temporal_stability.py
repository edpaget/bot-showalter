from dataclasses import dataclass


@dataclass(frozen=True)
class TargetStability:
    target: str
    per_season_r: tuple[tuple[int, float], ...]
    mean_r: float
    std_r: float
    cv: float
    classification: str


@dataclass(frozen=True)
class StabilityResult:
    column_spec: str
    player_type: str
    seasons: tuple[int, ...]
    target_stabilities: tuple[TargetStability, ...]
