from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class TargetCorrelation:
    target: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    n: int


@dataclass(frozen=True)
class SeasonCorrelationResult:
    column_spec: str
    season: int
    player_type: PlayerType
    correlations: tuple[TargetCorrelation, ...]


@dataclass(frozen=True)
class PooledCorrelationResult:
    column_spec: str
    player_type: PlayerType
    correlations: tuple[TargetCorrelation, ...]


@dataclass(frozen=True)
class CorrelationScanResult:
    column_spec: str
    player_type: PlayerType
    per_season: tuple[SeasonCorrelationResult, ...]
    pooled: PooledCorrelationResult


@dataclass(frozen=True)
class MultiColumnRanking:
    column_spec: str
    avg_abs_pearson: float
    avg_abs_spearman: float
