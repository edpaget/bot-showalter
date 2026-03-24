from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class TargetResult:
    rmse: float
    baseline_rmse: float
    delta: float
    delta_pct: float


@dataclass(frozen=True)
class FeatureExplorationResult:
    feature: str
    best_delta_pct: float
    best_experiment_id: int
    times_tested: int


@dataclass(frozen=True)
class TargetExplorationResult:
    target: str
    best_rmse: float
    best_delta_pct: float
    best_experiment_id: int
    experiments_count: int


@dataclass(frozen=True)
class ExplorationSummary:
    model: str
    player_type: PlayerType
    total_experiments: int
    features_tested: list[FeatureExplorationResult]
    targets_explored: list[TargetExplorationResult]
    best_experiment_id: int | None
    best_experiment_delta_pct: float | None


@dataclass(frozen=True)
class Experiment:
    timestamp: str
    hypothesis: str
    model: str
    player_type: PlayerType
    feature_diff: dict[str, list[str]]
    seasons: dict[str, list[int]]
    params: dict[str, Any]
    target_results: dict[str, TargetResult]
    conclusion: str
    tags: list[str] = field(default_factory=list)
    parent_id: int | None = None
    id: int | None = None
