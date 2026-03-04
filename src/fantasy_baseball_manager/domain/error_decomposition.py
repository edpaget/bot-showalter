from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, median, quantiles


@dataclass(frozen=True)
class PlayerResidual:
    player_id: int
    player_name: str
    predicted: float
    actual: float
    residual: float
    feature_values: dict[str, float]


@dataclass(frozen=True)
class DistinguishingFeature:
    feature_name: str
    mean_miss_group: float
    mean_rest: float
    difference: float


@dataclass(frozen=True)
class MissPopulationSummary:
    mean_age: float | None
    position_distribution: dict[str, int]
    mean_volume: float
    distinguishing_features: list[DistinguishingFeature]


@dataclass(frozen=True)
class ErrorDecompositionReport:
    target: str
    player_type: str
    season: int
    system: str
    version: str
    top_misses: list[PlayerResidual]
    over_predictions: list[PlayerResidual]
    under_predictions: list[PlayerResidual]
    summary: MissPopulationSummary


def rank_residuals(player_residuals: list[PlayerResidual], top_n: int) -> list[PlayerResidual]:
    """Sort by absolute residual descending, return top-N."""
    sorted_residuals = sorted(player_residuals, key=lambda r: abs(r.residual), reverse=True)
    return sorted_residuals[:top_n]


def split_direction(
    player_residuals: list[PlayerResidual],
) -> tuple[list[PlayerResidual], list[PlayerResidual]]:
    """Split into over-predictions (residual > 0) and under-predictions (residual < 0).

    Over-predictions are sorted by residual descending.
    Under-predictions are sorted by residual ascending (most negative first).
    """
    over = sorted([r for r in player_residuals if r.residual > 0], key=lambda r: r.residual, reverse=True)
    under = sorted([r for r in player_residuals if r.residual < 0], key=lambda r: r.residual)
    return over, under


def compute_distinguishing_features(
    miss_group: list[PlayerResidual],
    rest: list[PlayerResidual],
) -> list[DistinguishingFeature]:
    """For each numeric feature present in both groups, compute mean difference."""
    if not miss_group or not rest:
        return []

    miss_features: set[str] = set()
    for r in miss_group:
        miss_features.update(r.feature_values.keys())

    rest_features: set[str] = set()
    for r in rest:
        rest_features.update(r.feature_values.keys())

    common_features = miss_features & rest_features

    results: list[DistinguishingFeature] = []
    for feature in common_features:
        miss_vals = [r.feature_values[feature] for r in miss_group if feature in r.feature_values]
        rest_vals = [r.feature_values[feature] for r in rest if feature in r.feature_values]
        if not miss_vals or not rest_vals:
            continue
        miss_mean = mean(miss_vals)
        rest_mean = mean(rest_vals)
        diff = abs(miss_mean - rest_mean)
        results.append(
            DistinguishingFeature(
                feature_name=feature,
                mean_miss_group=miss_mean,
                mean_rest=rest_mean,
                difference=diff,
            )
        )

    results.sort(key=lambda f: f.difference, reverse=True)
    return results


def compute_miss_summary(
    misses: list[PlayerResidual],
    rest: list[PlayerResidual],
    player_positions: dict[int, str],
) -> MissPopulationSummary:
    """Compute summary statistics for the miss population."""
    age_values = [r.feature_values["age"] for r in misses if "age" in r.feature_values]
    mean_age = mean(age_values) if age_values else None

    position_distribution: dict[str, int] = {}
    for r in misses:
        pos = player_positions.get(r.player_id)
        if pos is not None:
            position_distribution[pos] = position_distribution.get(pos, 0) + 1

    volume_key = "pa"
    volume_values = [r.feature_values[volume_key] for r in misses if volume_key in r.feature_values]
    if not volume_values:
        volume_key = "ip"
        volume_values = [r.feature_values[volume_key] for r in misses if volume_key in r.feature_values]
    mean_volume = mean(volume_values) if volume_values else 0.0

    distinguishing_features = compute_distinguishing_features(misses, rest)

    return MissPopulationSummary(
        mean_age=mean_age,
        position_distribution=position_distribution,
        mean_volume=mean_volume,
        distinguishing_features=distinguishing_features,
    )


@dataclass(frozen=True)
class FeatureGap:
    feature_name: str
    ks_statistic: float
    p_value: float
    mean_well: float
    mean_poor: float
    in_model: bool


@dataclass(frozen=True)
class FeatureGapReport:
    target: str
    player_type: str
    season: int
    system: str
    version: str
    gaps: list[FeatureGap] = field(default_factory=list)


def split_residuals_by_quality(
    residuals: list[PlayerResidual],
    miss_percentile: float = 80.0,
) -> tuple[list[PlayerResidual], list[PlayerResidual]]:
    """Split into well-predicted and poorly-predicted groups.

    Well-predicted: absolute residual strictly below the median.
    Poorly-predicted: absolute residual strictly above the ``miss_percentile`` threshold.
    """
    if len(residuals) < 2:
        return [], []

    abs_residuals = [abs(r.residual) for r in residuals]
    med = median(abs_residuals)

    # quantiles with n-1 cuts gives percentiles at 1/n, 2/n, ... positions.
    # For miss_percentile, we need to translate to the right quantile cut.
    n = 100
    cuts = quantiles(abs_residuals, n=n)
    # cuts has n-1 values (indices 0..98 for n=100), representing 1st..99th percentiles
    idx = max(0, min(int(miss_percentile) - 1, len(cuts) - 1))
    threshold = cuts[idx]

    well = [r for r in residuals if abs(r.residual) < med]
    poor = [r for r in residuals if abs(r.residual) > threshold]

    return well, poor


@dataclass(frozen=True)
class CohortBias:
    cohort_label: str
    n: int
    mean_residual: float
    mean_abs_residual: float
    rmse: float
    significant: bool


@dataclass(frozen=True)
class CohortBiasReport:
    target: str
    player_type: str
    season: int
    system: str
    version: str
    dimension: str
    cohorts: list[CohortBias]


_AGE_BUCKETS = [("22-25", 22, 25), ("26-29", 26, 29), ("30-33", 30, 33), ("34+", 34, 999)]


def bucket_by_age(residuals: list[PlayerResidual]) -> dict[str, list[PlayerResidual]]:
    """Group by age bucket using feature_values['age']."""
    if not residuals:
        return {}
    result: dict[str, list[PlayerResidual]] = {}
    for r in residuals:
        age = r.feature_values.get("age")
        if age is None:
            continue
        age_int = int(age)
        for label, lo, hi in _AGE_BUCKETS:
            if lo <= age_int <= hi:
                result.setdefault(label, []).append(r)
                break
    return result


def bucket_by_position(
    residuals: list[PlayerResidual],
    primary_positions: dict[int, str],
) -> dict[str, list[PlayerResidual]]:
    """Group by primary position."""
    if not residuals:
        return {}
    result: dict[str, list[PlayerResidual]] = {}
    for r in residuals:
        pos = primary_positions.get(r.player_id)
        if pos is None:
            continue
        result.setdefault(pos, []).append(r)
    return result


def bucket_by_handedness(
    residuals: list[PlayerResidual],
    handedness: dict[int, str],
) -> dict[str, list[PlayerResidual]]:
    """Group by batting/throwing hand. Expects decoded strings (L/R/S)."""
    if not residuals:
        return {}
    result: dict[str, list[PlayerResidual]] = {}
    for r in residuals:
        hand = handedness.get(r.player_id)
        if hand is None:
            continue
        result.setdefault(hand, []).append(r)
    return result


_EXPERIENCE_BUCKETS = [("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, 999)]


def bucket_by_experience(
    residuals: list[PlayerResidual],
    experience: dict[int, int],
) -> dict[str, list[PlayerResidual]]:
    """Group by years of MLB experience."""
    if not residuals:
        return {}
    result: dict[str, list[PlayerResidual]] = {}
    for r in residuals:
        years = experience.get(r.player_id)
        if years is None:
            continue
        for label, lo, hi in _EXPERIENCE_BUCKETS:
            if lo <= years <= hi:
                result.setdefault(label, []).append(r)
                break
    return result


def compute_cohort_metrics(residuals: list[PlayerResidual]) -> tuple[float, float, float]:
    """Return (mean_residual, mean_abs_residual, rmse) for a group."""
    vals = [r.residual for r in residuals]
    mean_residual = mean(vals)
    mean_abs_residual = mean(abs(v) for v in vals)
    rmse = math.sqrt(mean(v * v for v in vals))
    return mean_residual, mean_abs_residual, rmse
