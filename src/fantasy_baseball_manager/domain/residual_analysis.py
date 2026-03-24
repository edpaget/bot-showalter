from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain.talent_quality import _safe_correlation

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain.identity import PlayerType


@dataclass(frozen=True)
class CalibrationBin:
    bin_center: float
    mean_predicted: float
    mean_actual: float
    mean_residual: float
    count: int


@dataclass(frozen=True)
class StatResidualAnalysis:
    stat_name: str
    player_type: PlayerType
    n_observations: int
    mean_residual: float
    std_residual: float
    bias_significant: bool
    heteroscedasticity_corr: float
    heteroscedasticity_significant: bool
    calibration_bins: list[CalibrationBin]


@dataclass(frozen=True)
class ResidualAnalysisSummary:
    n_bias_significant: int
    n_bias_total: int
    n_hetero_significant: int
    n_hetero_total: int
    calibration_recommended: bool


@dataclass(frozen=True)
class ResidualAnalysisReport:
    system: str
    version: str
    seasons: list[int]
    top: int | None
    stat_analyses: list[StatResidualAnalysis]
    summary: ResidualAnalysisSummary


def compute_mean_bias(residuals: list[float]) -> tuple[float, bool]:
    """Compute mean residual and whether the bias is statistically significant.

    Uses a t-test approximation: significant if |t| > 2.0 (fine for n > 50).
    Returns (mean, is_significant).
    """
    n = len(residuals)
    if n < 2:
        return 0.0, False

    mean = statistics.mean(residuals)

    try:
        stdev = statistics.stdev(residuals)
    except statistics.StatisticsError:
        return mean, False

    if stdev == 0:
        return mean, False

    t_stat = mean / (stdev / math.sqrt(n))
    return mean, abs(t_stat) > 2.0


def compute_heteroscedasticity(
    predictions: list[float],
    abs_residuals: list[float],
) -> tuple[float, bool]:
    """Compute correlation between |residual| and predicted value.

    Returns (correlation, is_significant). Significant if |r| > 0.15.
    """
    corr = _safe_correlation(predictions, abs_residuals)
    return corr, abs(corr) > 0.15


def compute_calibration_bins(
    predictions: list[float],
    actuals: list[float],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Compute equal-count calibration bins sorted by predicted value.

    Returns a list of CalibrationBin sorted by bin_center.
    """
    n = len(predictions)
    if n == 0:
        return []

    # Sort by predicted value
    paired = sorted(zip(predictions, actuals, strict=True), key=lambda x: x[0])

    # Split into approximately equal-count bins
    actual_bins = min(n_bins, n)
    base_size = n // actual_bins
    remainder = n % actual_bins

    bins: list[CalibrationBin] = []
    idx = 0
    for i in range(actual_bins):
        size = base_size + (1 if i < remainder else 0)
        chunk = paired[idx : idx + size]
        idx += size

        preds = [p for p, _ in chunk]
        acts = [a for _, a in chunk]
        mean_pred = statistics.mean(preds)
        mean_act = statistics.mean(acts)
        bin_center = (preds[0] + preds[-1]) / 2

        bins.append(
            CalibrationBin(
                bin_center=bin_center,
                mean_predicted=mean_pred,
                mean_actual=mean_act,
                mean_residual=mean_act - mean_pred,
                count=size,
            )
        )

    return bins
