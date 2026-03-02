from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetResult:
    rmse: float
    baseline_rmse: float
    delta: float
    delta_pct: float


@dataclass(frozen=True)
class Experiment:
    timestamp: str
    hypothesis: str
    model: str
    player_type: str
    feature_diff: dict[str, list[str]]
    seasons: dict[str, list[int]]
    params: dict[str, Any]
    target_results: dict[str, TargetResult]
    conclusion: str
    tags: list[str] = field(default_factory=list)
    parent_id: int | None = None
    id: int | None = None
