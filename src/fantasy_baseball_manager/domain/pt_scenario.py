from dataclasses import dataclass


@dataclass(frozen=True)
class PlayingTimeScenario:
    """A single playing-time scenario with its probability weight."""

    percentile: int  # e.g. 10, 25, 50, 75, 90
    pa_or_ip: float  # the PT value for this scenario
    weight: float  # probability mass (sums to 1.0 across scenarios)
