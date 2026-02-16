from fantasy_baseball_manager.models.mle.types import AgeAdjustmentConfig


def compute_age_adjustment(*, age: float, level: str, config: AgeAdjustmentConfig) -> float:
    """Compute an age-for-level multiplier for MLE translation.

    Returns a multiplier (clamped to [min_multiplier, max_multiplier]) that
    adjusts translated rates based on how the player's age compares to the
    optimal benchmark for his level, plus a development projection toward
    peak age.
    """
    benchmark = config.benchmarks.get(level, age)
    age_diff = benchmark - age  # positive = young for level

    if age_diff > 0:
        age_component = age_diff * config.young_bonus_per_year
    else:
        age_component = age_diff * config.old_penalty_per_year

    years_to_peak = max(0.0, config.peak_age - age)
    dev_bonus = years_to_peak * config.development_rate_per_year

    raw = 1.0 + age_component + dev_bonus
    return max(config.min_multiplier, min(raw, config.max_multiplier))
