import pytest

from fantasy_baseball_manager.models.mle.age_adjustment import compute_age_adjustment
from fantasy_baseball_manager.models.mle.types import (
    AgeAdjustmentConfig,
    DEFAULT_AGE_BENCHMARKS,
)


def _default_config() -> AgeAdjustmentConfig:
    return AgeAdjustmentConfig(benchmarks=DEFAULT_AGE_BENCHMARKS)


class TestComputeAgeAdjustment:
    def test_at_benchmark_age(self) -> None:
        # 21-year-old at AA: age_diff=0, dev projection only
        result = compute_age_adjustment(age=21.0, level="AA", config=_default_config())
        # dev bonus = (27 - 21) * 0.006 = 0.036 → multiplier = 1.036
        assert result == pytest.approx(1.036)

    def test_young_for_level(self) -> None:
        # 20 at AA (1 year young) → higher than 21 at AA
        young = compute_age_adjustment(age=20.0, level="AA", config=_default_config())
        benchmark = compute_age_adjustment(age=21.0, level="AA", config=_default_config())
        assert young > benchmark

    def test_old_for_level(self) -> None:
        # 24 at AA (3 years old)
        # age-for-level: (21-24) * 0.010 = -0.030
        # dev projection: (27-24) * 0.006 = 0.018
        # total = 1.0 - 0.030 + 0.018 = 0.988
        result = compute_age_adjustment(age=24.0, level="AA", config=_default_config())
        assert result == pytest.approx(0.988)
        assert result < 1.0

    def test_asymmetric(self) -> None:
        # 1 year young bonus > 1 year old penalty (in absolute magnitude)
        young = compute_age_adjustment(age=20.0, level="AA", config=_default_config())
        old = compute_age_adjustment(age=22.0, level="AA", config=_default_config())
        benchmark = compute_age_adjustment(age=21.0, level="AA", config=_default_config())
        young_bonus = young - benchmark
        old_penalty = benchmark - old
        assert young_bonus > old_penalty

    def test_bounded_max(self) -> None:
        # Very young player at a high level → capped at max_multiplier
        result = compute_age_adjustment(age=14.0, level="AAA", config=_default_config())
        assert result == pytest.approx(1.25)

    def test_bounded_min(self) -> None:
        # Very old player at a low level → capped at min_multiplier
        result = compute_age_adjustment(age=35.0, level="A", config=_default_config())
        assert result == pytest.approx(0.85)

    def test_unknown_level_defaults_to_no_bonus(self) -> None:
        # Unknown level: benchmark = age → no age-for-level, dev projection still applies
        result = compute_age_adjustment(age=22.0, level="UNKNOWN", config=_default_config())
        # dev bonus = (27 - 22) * 0.006 = 0.030 → multiplier = 1.030
        assert result == pytest.approx(1.030)

    def test_development_projection_zero_at_peak(self) -> None:
        # Player at peak age → 0 dev bonus
        result = compute_age_adjustment(age=27.0, level="AAA", config=_default_config())
        # age-for-level: (22-27) * 0.010 = -0.050
        # dev bonus: 0
        # total = 1.0 - 0.050 = 0.950
        assert result == pytest.approx(0.950)

    def test_development_projection_increases_for_younger(self) -> None:
        young = compute_age_adjustment(age=20.0, level="AA", config=_default_config())
        older = compute_age_adjustment(age=25.0, level="AA", config=_default_config())
        assert young > older
