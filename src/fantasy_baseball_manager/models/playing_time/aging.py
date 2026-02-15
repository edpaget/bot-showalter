"""Aging curve for playing-time projection."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgingCurve:
    peak_age: float
    improvement_rate: float  # per-year fractional improvement before peak (positive)
    decline_rate: float  # per-year fractional decline after peak (positive)
    player_type: str  # "batter" or "pitcher"


def compute_age_pt_factor(age: float | int | None, curve: AgingCurve) -> float:
    """Piecewise-linear multiplier centered at 1.0 at peak_age."""
    if age is None:
        return 1.0
    diff = curve.peak_age - float(age)
    if diff > 0:
        return 1.0 + diff * curve.improvement_rate
    elif diff < 0:
        return 1.0 + diff * curve.decline_rate  # diff is negative
    return 1.0


_DEFAULT_BATTER_CURVE = AgingCurve(
    peak_age=27.0,
    improvement_rate=0.01,
    decline_rate=0.005,
    player_type="batter",
)
_DEFAULT_PITCHER_CURVE = AgingCurve(
    peak_age=26.0,
    improvement_rate=0.008,
    decline_rate=0.007,
    player_type="pitcher",
)


def fit_playing_time_aging_curve(
    rows: list[dict[str, Any]],
    player_type: str,
    current_column: str,
    prior_column: str,
    min_pt: float = 50.0,
    min_samples: int = 30,
) -> AgingCurve:
    """Fit an aging curve from proportional playing-time deltas using the delta method."""
    default = _DEFAULT_BATTER_CURVE if player_type == "batter" else _DEFAULT_PITCHER_CURVE

    # Compute proportional delta per row, group by integer age
    age_deltas: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        age = row.get("age")
        if age is None:
            continue
        prior = row.get(prior_column)
        current = row.get(current_column)
        if prior is None or current is None:
            continue
        prior_val = float(prior)
        if prior_val <= 0 or prior_val < min_pt:
            continue
        delta = (float(current) - prior_val) / prior_val
        age_deltas[int(age)].append(delta)

    # Filter ages with insufficient samples and compute mean delta per age
    mean_deltas: dict[int, float] = {}
    for age, deltas in age_deltas.items():
        if len(deltas) >= min_samples:
            mean_deltas[age] = sum(deltas) / len(deltas)

    if len(mean_deltas) < 2:
        return AgingCurve(
            peak_age=default.peak_age,
            improvement_rate=default.improvement_rate,
            decline_rate=default.decline_rate,
            player_type=player_type,
        )

    # Find peak_age: age where cumulative sum of mean deltas is maximized
    sorted_ages = sorted(mean_deltas.keys())
    cum_sum = 0.0
    max_cum = float("-inf")
    peak_age = sorted_ages[0]
    for age in sorted_ages:
        cum_sum += mean_deltas[age]
        if cum_sum >= max_cum:
            max_cum = cum_sum
            peak_age = age

    # Compute improvement rate (mean of deltas for ages below peak)
    below_peak = [mean_deltas[a] for a in sorted_ages if a < peak_age]
    improvement_rate = sum(below_peak) / len(below_peak) if below_peak else 0.0

    # Compute decline rate (negative mean of deltas for ages above peak)
    above_peak = [mean_deltas[a] for a in sorted_ages if a > peak_age]
    decline_rate = -(sum(above_peak) / len(above_peak)) if above_peak else 0.0

    # Clamp to non-negative
    improvement_rate = max(0.0, improvement_rate)
    decline_rate = max(0.0, decline_rate)

    return AgingCurve(
        peak_age=float(peak_age),
        improvement_rate=improvement_rate,
        decline_rate=decline_rate,
        player_type=player_type,
    )


def enrich_rows_with_age_pt_factor(
    rows: list[dict[str, Any]],
    curve: AgingCurve,
) -> list[dict[str, Any]]:
    """Return new list of rows with 'age_pt_factor' added (non-mutating)."""
    return [{**row, "age_pt_factor": compute_age_pt_factor(row.get("age"), curve)} for row in rows]
