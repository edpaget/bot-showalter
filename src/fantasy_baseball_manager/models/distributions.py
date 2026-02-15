from collections.abc import Sequence

import numpy as np

from fantasy_baseball_manager.domain.projection import StatDistribution


def samples_to_distribution(stat: str, samples: Sequence[float]) -> StatDistribution:
    """Convert Monte Carlo samples to a StatDistribution.

    Computes p10/p25/p50/p75/p90 from empirical quantiles via numpy.
    Computes mean and std from the sample.  Does not fit a parametric family
    (``family`` field left as ``None``).

    Raises ``ValueError`` if fewer than 2 samples are provided.
    """
    if len(samples) < 2:
        msg = f"Need at least 2 samples, got {len(samples)}"
        raise ValueError(msg)

    arr = np.asarray(samples, dtype=np.float64)
    p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])

    return StatDistribution(
        stat=stat,
        p10=float(p10),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p90=float(p90),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
    )


def samples_to_distributions(
    stat_samples: dict[str, Sequence[float]],
) -> dict[str, StatDistribution]:
    """Batch version: convert samples for multiple stats."""
    return {stat: samples_to_distribution(stat, samples) for stat, samples in stat_samples.items()}
