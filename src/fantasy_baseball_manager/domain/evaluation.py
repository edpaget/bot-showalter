import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from fantasy_baseball_manager.domain.projection_accuracy import ProjectionComparison


@dataclass(frozen=True)
class StatMetrics:
    rmse: float
    mae: float
    correlation: float
    n: int


@dataclass(frozen=True)
class SystemMetrics:
    system: str
    version: str
    source_type: str
    metrics: dict[str, StatMetrics]


@dataclass(frozen=True)
class ComparisonResult:
    season: int
    stats: list[str]
    systems: list[SystemMetrics]


@dataclass(frozen=True)
class StratifiedComparisonResult:
    dimension: str
    season: int
    stats: list[str]
    cohorts: dict[str, ComparisonResult]


def compute_stat_metrics(
    comparisons: list[ProjectionComparison],
    stats: list[str] | None = None,
) -> dict[str, StatMetrics]:
    """Aggregate per-player comparisons into per-stat RMSE/MAE/correlation."""
    by_stat: dict[str, list[ProjectionComparison]] = defaultdict(list)
    for comp in comparisons:
        if stats is not None and comp.stat_name not in stats:
            continue
        by_stat[comp.stat_name].append(comp)

    result: dict[str, StatMetrics] = {}
    for stat_name, comps in by_stat.items():
        n = len(comps)
        errors = [c.error for c in comps]
        mae = sum(abs(e) for e in errors) / n
        rmse = math.sqrt(sum(e * e for e in errors) / n)

        if n < 2:
            correlation = 0.0
        else:
            projected = [c.projected for c in comps]
            actual = [c.actual for c in comps]
            try:
                correlation = statistics.correlation(projected, actual)
            except statistics.StatisticsError:
                correlation = 0.0

        result[stat_name] = StatMetrics(rmse=rmse, mae=mae, correlation=correlation, n=n)

    return result
