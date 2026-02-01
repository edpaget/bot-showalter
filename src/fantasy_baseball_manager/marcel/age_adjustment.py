PEAK_AGE = 29
YOUNG_RATE = 0.006
OLD_RATE = 0.003


def age_multiplier(age: int) -> float:
    """Return the MARCEL age adjustment multiplier.

    Players younger than 29 get a boost (+0.6%/year),
    players older than 29 get a penalty (-0.3%/year).
    """
    if age < PEAK_AGE:
        return 1.0 + (PEAK_AGE - age) * YOUNG_RATE
    elif age > PEAK_AGE:
        return 1.0 + (PEAK_AGE - age) * OLD_RATE
    return 1.0
