from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


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
