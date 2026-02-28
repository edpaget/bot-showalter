import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.projection_accuracy import ProjectionComparison


@dataclass(frozen=True)
class StatMetrics:
    rmse: float
    mae: float
    correlation: float
    r_squared: float
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
class StatComparisonRecord:
    stat_name: str
    baseline_rmse: float
    candidate_rmse: float
    rmse_delta: float
    rmse_pct_delta: float
    rmse_winner: str
    baseline_r_squared: float
    candidate_r_squared: float
    r_squared_delta: float
    r_squared_pct_delta: float
    r_squared_winner: str


@dataclass(frozen=True)
class ComparisonSummary:
    baseline_label: str
    candidate_label: str
    records: list[StatComparisonRecord]
    rmse_wins: int
    rmse_losses: int
    rmse_ties: int
    r_squared_wins: int
    r_squared_losses: int
    r_squared_ties: int


@dataclass(frozen=True)
class StratifiedComparisonResult:
    dimension: str
    season: int
    stats: list[str]
    cohorts: dict[str, ComparisonResult]


_TIE_TOLERANCE = 1e-6


def _determine_winner(delta: float, lower_is_better: bool) -> str:
    if abs(delta) < _TIE_TOLERANCE:
        return "tie"
    if lower_is_better:
        return "candidate" if delta < 0 else "baseline"
    return "candidate" if delta > 0 else "baseline"


def _pct_delta(delta: float, baseline: float) -> float:
    if abs(baseline) < _TIE_TOLERANCE:
        return 0.0
    return (delta / abs(baseline)) * 100.0


def summarize_comparison(
    result: ComparisonResult,
    baseline_index: int = 0,
    candidate_index: int = 1,
) -> ComparisonSummary:
    """Build a comparison summary with deltas and win/loss tallies."""
    baseline = result.systems[baseline_index]
    candidate = result.systems[candidate_index]

    baseline_label = f"{baseline.system}/{baseline.version}"
    candidate_label = f"{candidate.system}/{candidate.version}"

    records: list[StatComparisonRecord] = []
    rmse_wins = rmse_losses = rmse_ties = 0
    r_squared_wins = r_squared_losses = r_squared_ties = 0

    for stat_name in result.stats:
        b_metrics = baseline.metrics.get(stat_name)
        c_metrics = candidate.metrics.get(stat_name)
        if b_metrics is None or c_metrics is None:
            continue

        rmse_delta = c_metrics.rmse - b_metrics.rmse
        rmse_winner = _determine_winner(rmse_delta, lower_is_better=True)

        r_sq_delta = c_metrics.r_squared - b_metrics.r_squared
        r_sq_winner = _determine_winner(r_sq_delta, lower_is_better=False)

        records.append(
            StatComparisonRecord(
                stat_name=stat_name,
                baseline_rmse=b_metrics.rmse,
                candidate_rmse=c_metrics.rmse,
                rmse_delta=rmse_delta,
                rmse_pct_delta=_pct_delta(rmse_delta, b_metrics.rmse),
                rmse_winner=rmse_winner,
                baseline_r_squared=b_metrics.r_squared,
                candidate_r_squared=c_metrics.r_squared,
                r_squared_delta=r_sq_delta,
                r_squared_pct_delta=_pct_delta(r_sq_delta, b_metrics.r_squared),
                r_squared_winner=r_sq_winner,
            )
        )

        if rmse_winner == "candidate":
            rmse_wins += 1
        elif rmse_winner == "baseline":
            rmse_losses += 1
        else:
            rmse_ties += 1

        if r_sq_winner == "candidate":
            r_squared_wins += 1
        elif r_sq_winner == "baseline":
            r_squared_losses += 1
        else:
            r_squared_ties += 1

    return ComparisonSummary(
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        records=records,
        rmse_wins=rmse_wins,
        rmse_losses=rmse_losses,
        rmse_ties=rmse_ties,
        r_squared_wins=r_squared_wins,
        r_squared_losses=r_squared_losses,
        r_squared_ties=r_squared_ties,
    )


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

        ss_res = sum(e * e for e in errors)
        mean_actual = sum(c.actual for c in comps) / n
        ss_tot = sum((c.actual - mean_actual) ** 2 for c in comps)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result[stat_name] = StatMetrics(rmse=rmse, mae=mae, correlation=correlation, r_squared=r_squared, n=n)

    return result
