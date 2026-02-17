import math
import statistics
from dataclasses import dataclass

from fantasy_baseball_manager.domain.talent_quality import _safe_correlation


@dataclass(frozen=True)
class ChronicPerformer:
    player_id: int
    player_name: str
    residual_n: float
    residual_n1: float
    mean_residual: float
    pa_n: float
    pa_n1: float


@dataclass(frozen=True)
class StatResidualPersistence:
    stat_name: str
    residual_corr_overall: float
    residual_corr_by_bucket: dict[str, float]
    n_by_bucket: dict[str, int]
    chronic_overperformers: list[ChronicPerformer]
    chronic_underperformers: list[ChronicPerformer]
    rmse_baseline: float
    rmse_corrected: float
    rmse_improvement_pct: float
    persistence_pass: bool
    ceiling_pass: bool
    n_returning: int


@dataclass(frozen=True)
class ResidualPersistenceSummary:
    persistence_passes: int
    persistence_total: int
    ceiling_passes: int
    ceiling_total: int
    go: bool


@dataclass(frozen=True)
class ResidualPersistenceReport:
    system: str
    version: str
    season_n: int
    season_n1: int
    stat_metrics: list[StatResidualPersistence]
    summary: ResidualPersistenceSummary


def compute_residual_correlation_by_bucket(
    residuals_n: list[float],
    residuals_n1: list[float],
    sample_sizes: list[float],
    bucket_edges: tuple[float, ...],
    bucket_labels: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute Pearson r of residuals within each PA bucket.

    Returns (corr_by_bucket, n_by_bucket).
    """
    buckets_n: dict[str, list[float]] = {}
    buckets_n1: dict[str, list[float]] = {}

    for i in range(len(residuals_n)):
        ss = sample_sizes[i]
        label = bucket_labels[-1]
        for j, edge in enumerate(bucket_edges):
            if ss < edge:
                label = bucket_labels[j]
                break
        buckets_n.setdefault(label, []).append(residuals_n[i])
        buckets_n1.setdefault(label, []).append(residuals_n1[i])

    corr_by_bucket: dict[str, float] = {}
    n_by_bucket: dict[str, int] = {}

    for label in buckets_n:
        n_by_bucket[label] = len(buckets_n[label])
        corr_by_bucket[label] = _safe_correlation(buckets_n[label], buckets_n1[label])

    return corr_by_bucket, n_by_bucket


def identify_chronic_performers(
    residuals_n: list[float],
    residuals_n1: list[float],
    player_ids: list[int],
    player_names: list[str],
    pa_n: list[float],
    pa_n1: list[float],
) -> tuple[list[ChronicPerformer], list[ChronicPerformer]]:
    """Identify players with residuals > +1σ or < -1σ in both seasons.

    Returns (overperformers, underperformers) sorted by abs(mean_residual) descending.
    """
    if len(residuals_n) < 2:
        return [], []

    try:
        stdev_n = statistics.stdev(residuals_n)
    except statistics.StatisticsError:
        return [], []

    if stdev_n == 0:
        return [], []

    overperformers: list[ChronicPerformer] = []
    underperformers: list[ChronicPerformer] = []

    for i in range(len(residuals_n)):
        r_n = residuals_n[i]
        r_n1 = residuals_n1[i]
        mean_r = (r_n + r_n1) / 2

        if r_n > stdev_n and r_n1 > stdev_n:
            overperformers.append(
                ChronicPerformer(
                    player_id=player_ids[i],
                    player_name=player_names[i],
                    residual_n=r_n,
                    residual_n1=r_n1,
                    mean_residual=mean_r,
                    pa_n=pa_n[i],
                    pa_n1=pa_n1[i],
                )
            )
        elif r_n < -stdev_n and r_n1 < -stdev_n:
            underperformers.append(
                ChronicPerformer(
                    player_id=player_ids[i],
                    player_name=player_names[i],
                    residual_n=r_n,
                    residual_n1=r_n1,
                    mean_residual=mean_r,
                    pa_n=pa_n[i],
                    pa_n1=pa_n1[i],
                )
            )

    overperformers.sort(key=lambda p: abs(p.mean_residual), reverse=True)
    underperformers.sort(key=lambda p: abs(p.mean_residual), reverse=True)

    return overperformers, underperformers


def compute_rmse_ceiling(
    actuals_n1: list[float],
    model_estimates_n1: list[float],
    mean_prior_residuals: list[float],
) -> tuple[float, float, float]:
    """Compute RMSE improvement ceiling from applying prior residuals.

    Returns (rmse_baseline, rmse_corrected, improvement_pct).
    """
    n = len(actuals_n1)
    if n == 0:
        return 0.0, 0.0, 0.0

    ss_baseline = sum((actuals_n1[i] - model_estimates_n1[i]) ** 2 for i in range(n))
    ss_corrected = sum((actuals_n1[i] - (model_estimates_n1[i] + mean_prior_residuals[i])) ** 2 for i in range(n))

    rmse_baseline = math.sqrt(ss_baseline / n)
    rmse_corrected = math.sqrt(ss_corrected / n)

    if rmse_baseline == 0:
        return 0.0, 0.0, 0.0

    improvement_pct = (rmse_baseline - rmse_corrected) / rmse_baseline * 100

    return rmse_baseline, rmse_corrected, improvement_pct
