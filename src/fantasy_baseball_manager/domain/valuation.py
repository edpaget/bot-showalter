from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Valuation:
    player_id: int
    season: int
    system: str
    version: str
    projection_system: str
    projection_version: str
    player_type: str
    position: str
    value: float
    rank: int
    category_scores: dict[str, float]
    id: int | None = None
    loaded_at: str | None = None


@dataclass(frozen=True)
class PlayerValuation:
    player_name: str
    system: str
    version: str
    projection_system: str
    projection_version: str
    player_type: str
    position: str
    value: float
    rank: int
    category_scores: dict[str, float]


@dataclass(frozen=True)
class ValuationAccuracy:
    player_id: int
    player_name: str
    player_type: str
    position: str
    predicted_value: float
    actual_value: float
    surplus: float  # predicted - actual (positive = overpaid)
    predicted_rank: int
    actual_rank: int
    actual_war: float | None = None


@dataclass(frozen=True)
class ValuationEvalResult:
    system: str
    version: str
    season: int
    value_mae: float
    rank_correlation: float  # Spearman rho
    n: int
    players: list[ValuationAccuracy]
    total_matched: int | None = None
    filter_description: str | None = None
    war_correlation: float | None = None
    war_correlation_batters: float | None = None
    war_correlation_pitchers: float | None = None
    hit_rates: dict[int, float] | None = None
    cohorts: dict[str, ValuationEvalResult] | None = None
    tail_results: dict[int, ValuationEvalResult] | None = None


@dataclass(frozen=True)
class ValuationComparisonResult:
    season: int
    baseline: ValuationEvalResult
    candidate: ValuationEvalResult


@dataclass(frozen=True)
class ValuationRegressionCheck:
    passed: bool
    war_passed: bool
    hit_rate_passed: bool
    explanation: str


def check_valuation_regression(
    baseline: ValuationEvalResult,
    candidate: ValuationEvalResult,
) -> ValuationRegressionCheck:
    """Check whether the candidate regresses on independent targets.

    Gates on WAR ρ (must not drop > 0.01) and average hit rate (must not drop > 5pp).
    Deliberately ignores ZAR$ metrics which are circular.
    """
    # WAR ρ check
    war_passed = True
    war_parts: list[str] = []
    if baseline.war_correlation is not None and candidate.war_correlation is not None:
        war_drop = baseline.war_correlation - candidate.war_correlation
        war_passed = war_drop <= 0.01 + 1e-9
        war_parts.append(
            f"WAR ρ {candidate.war_correlation:.4f} vs {baseline.war_correlation:.4f} "
            f"(Δ{candidate.war_correlation - baseline.war_correlation:+.4f})"
        )
    else:
        war_parts.append("WAR ρ: no data")

    # Hit rate check
    hit_rate_passed = True
    hr_parts: list[str] = []
    if baseline.hit_rates and candidate.hit_rates:
        common_ns = sorted(set(baseline.hit_rates) & set(candidate.hit_rates))
        if common_ns:
            baseline_avg = statistics.mean(baseline.hit_rates[n] for n in common_ns)
            candidate_avg = statistics.mean(candidate.hit_rates[n] for n in common_ns)
            drop = baseline_avg - candidate_avg
            hit_rate_passed = drop <= 5.0 + 1e-9
            hr_parts.append(
                f"Hit rate avg {candidate_avg:.1f}% vs {baseline_avg:.1f}% (Δ{candidate_avg - baseline_avg:+.1f}pp)"
            )
        else:
            hr_parts.append("Hit rate: no common Ns")
    else:
        hr_parts.append("Hit rate: no data")

    passed = war_passed and hit_rate_passed
    verdict = "PASS" if passed else "FAIL"
    explanation = f"{verdict}: {', '.join(war_parts + hr_parts)}"

    return ValuationRegressionCheck(
        passed=passed,
        war_passed=war_passed,
        hit_rate_passed=hit_rate_passed,
        explanation=explanation,
    )
