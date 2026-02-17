import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class StatTalentMetrics:
    stat_name: str
    # Next-season predictive validity
    model_next_season_corr: float
    raw_next_season_corr: float
    predictive_validity_pass: bool
    # Residual non-persistence
    residual_yoy_corr: float
    residual_non_persistence_pass: bool
    # Shrinkage quality
    shrinkage_ratio: float
    estimate_raw_corr: float
    shrinkage_pass: bool
    # R-squared decomposition
    r_squared: float
    residual_by_bucket: dict[str, float]
    r_squared_pass: bool
    # Residual regression rate
    regression_rate: float
    regression_rate_pass: bool
    # Sample sizes
    n_season_n: int
    n_returning: int


@dataclass(frozen=True)
class TalentQualitySummary:
    predictive_validity_passes: int
    predictive_validity_total: int
    residual_non_persistence_passes: int
    residual_non_persistence_total: int
    shrinkage_passes: int
    shrinkage_total: int
    r_squared_passes: int
    r_squared_total: int
    regression_rate_passes: int
    regression_rate_total: int


@dataclass(frozen=True)
class TrueTalentQualityReport:
    system: str
    version: str
    season_n: int
    season_n1: int
    player_type: str
    stat_metrics: list[StatTalentMetrics]
    summary: TalentQualitySummary


def _safe_correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return 0.0


def compute_predictive_validity(
    model_estimates_n: list[float],
    raw_actuals_n: list[float],
    actuals_n1: list[float],
) -> tuple[float, float]:
    """Return (model_corr_with_n1, raw_corr_with_n1)."""
    model_corr = _safe_correlation(model_estimates_n, actuals_n1)
    raw_corr = _safe_correlation(raw_actuals_n, actuals_n1)
    return model_corr, raw_corr


def compute_residual_yoy_correlation(
    residuals_n: list[float],
    residuals_n1: list[float],
) -> float:
    """Pearson correlation of residuals across seasons."""
    return _safe_correlation(residuals_n, residuals_n1)


def compute_shrinkage(
    estimates: list[float],
    raw_stats: list[float],
) -> tuple[float, float]:
    """Return (var_ratio, estimate_raw_correlation)."""
    if len(estimates) < 2:
        return 0.0, 0.0
    try:
        var_raw = statistics.variance(raw_stats)
    except statistics.StatisticsError:
        return 0.0, 0.0
    if var_raw == 0:
        return 0.0, _safe_correlation(estimates, raw_stats)
    var_est = statistics.variance(estimates)
    corr = _safe_correlation(estimates, raw_stats)
    return var_est / var_raw, corr


def compute_r_squared_with_buckets(
    estimates: list[float],
    actuals: list[float],
    sample_sizes: list[float],
    bucket_edges: tuple[float, ...],
    bucket_labels: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    """Return (r_squared, {bucket_label: mean_abs_residual})."""
    n = len(estimates)
    if n < 2:
        return 0.0, {}

    mean_actual = sum(actuals) / n
    ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
    if ss_tot == 0:
        return 0.0, {}

    residuals = [actuals[i] - estimates[i] for i in range(n)]
    ss_res = sum(r * r for r in residuals)
    r_squared = 1.0 - ss_res / ss_tot

    # Bucket assignment
    bucket_residuals: dict[str, list[float]] = {}
    for i in range(n):
        ss = sample_sizes[i]
        label = bucket_labels[-1]  # default to last bucket
        for j, edge in enumerate(bucket_edges):
            if ss < edge:
                label = bucket_labels[j]
                break
        bucket_residuals.setdefault(label, []).append(abs(residuals[i]))

    residual_by_bucket = {label: sum(vals) / len(vals) for label, vals in bucket_residuals.items()}

    return r_squared, residual_by_bucket
