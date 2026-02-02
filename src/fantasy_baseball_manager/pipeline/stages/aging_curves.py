"""Per-stat aging curve parameters for component-specific aging adjustments.

Each stat has its own peak age and aging rates. The multiplier formula is:
- At peak age: 1.0 (no adjustment)
- Younger than peak: 1.0 + (peak_age - age) * young_rate  (boost)
- Older than peak: 1.0 - (age - peak_age) * old_rate * position_modifier  (penalty)

For pitching inverted stats (where lower is better), the multiplier is inverted.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgingCurveParams:
    peak_age: int
    young_rate: float
    old_rate: float


BATTING_AGING_CURVES: dict[str, AgingCurveParams] = {
    # Speed stats: peak early, steep decline
    "sb": AgingCurveParams(peak_age=25, young_rate=0.008, old_rate=0.035),
    "triples": AgingCurveParams(peak_age=25, young_rate=0.008, old_rate=0.035),
    "cs": AgingCurveParams(peak_age=25, young_rate=0.008, old_rate=0.035),
    # Power: peak 28, moderate decline
    "hr": AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.015),
    # Contact: peak 27, moderate decline
    "singles": AgingCurveParams(peak_age=27, young_rate=0.005, old_rate=0.010),
    "doubles": AgingCurveParams(peak_age=27, young_rate=0.005, old_rate=0.012),
    # Discipline: peak 30, slow decline
    "bb": AgingCurveParams(peak_age=30, young_rate=0.004, old_rate=0.004),
    "so": AgingCurveParams(peak_age=30, young_rate=0.004, old_rate=0.004),
    # Composites: peak 28, moderate decline
    "r": AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.012),
    "rbi": AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.012),
    # Flat stats: peak 29, minimal aging
    "hbp": AgingCurveParams(peak_age=29, young_rate=0.002, old_rate=0.002),
    "sf": AgingCurveParams(peak_age=29, young_rate=0.002, old_rate=0.002),
    "sh": AgingCurveParams(peak_age=29, young_rate=0.002, old_rate=0.002),
}

PITCHING_AGING_CURVES: dict[str, AgingCurveParams] = {
    "so": AgingCurveParams(peak_age=26, young_rate=0.008, old_rate=0.015),
    "bb": AgingCurveParams(peak_age=28, young_rate=0.004, old_rate=0.006),
    "hr": AgingCurveParams(peak_age=27, young_rate=0.005, old_rate=0.010),
    "h": AgingCurveParams(peak_age=27, young_rate=0.005, old_rate=0.008),
    "hbp": AgingCurveParams(peak_age=29, young_rate=0.002, old_rate=0.002),
    "er": AgingCurveParams(peak_age=27, young_rate=0.005, old_rate=0.010),
    "w": AgingCurveParams(peak_age=28, young_rate=0.005, old_rate=0.010),
    "sv": AgingCurveParams(peak_age=28, young_rate=0.005, old_rate=0.010),
    "hld": AgingCurveParams(peak_age=28, young_rate=0.005, old_rate=0.010),
    "bs": AgingCurveParams(peak_age=28, young_rate=0.004, old_rate=0.008),
}

PITCHING_INVERTED_STATS: frozenset[str] = frozenset({"bb", "hr", "h", "hbp", "er", "bs"})

POSITION_AGING_MODIFIERS: dict[str, float] = {
    "C": 1.3,
    "IF": 1.0,
    "OF": 1.0,
    "SP": 1.0,
    "RP": 1.1,
    "DH": 0.9,
}
