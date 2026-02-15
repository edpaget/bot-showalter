"""Save/load playing-time coefficients and aging curves via joblib."""

from pathlib import Path

import joblib

from fantasy_baseball_manager.models.playing_time.aging import AgingCurve
from fantasy_baseball_manager.models.playing_time.engine import PlayingTimeCoefficients, ResidualBuckets


def save_coefficients(
    coefficients: dict[str, PlayingTimeCoefficients],
    path: Path,
) -> None:
    """Save batter/pitcher coefficients to a joblib file."""
    joblib.dump(coefficients, path)


def load_coefficients(path: Path) -> dict[str, PlayingTimeCoefficients]:
    """Load batter/pitcher coefficients from a joblib file."""
    if not path.exists():
        msg = f"Coefficients file not found: {path}"
        raise FileNotFoundError(msg)
    result: dict[str, PlayingTimeCoefficients] = joblib.load(path)
    return result


def save_aging_curves(
    curves: dict[str, AgingCurve],
    path: Path,
) -> None:
    """Save batter/pitcher aging curves to a joblib file."""
    joblib.dump(curves, path)


def load_aging_curves(path: Path) -> dict[str, AgingCurve]:
    """Load batter/pitcher aging curves from a joblib file."""
    if not path.exists():
        msg = f"Aging curves file not found: {path}"
        raise FileNotFoundError(msg)
    result: dict[str, AgingCurve] = joblib.load(path)
    return result


def save_residual_buckets(
    buckets: dict[str, ResidualBuckets],
    path: Path,
) -> None:
    """Save batter/pitcher residual buckets to a joblib file."""
    joblib.dump(buckets, path)


def load_residual_buckets(path: Path) -> dict[str, ResidualBuckets]:
    """Load batter/pitcher residual buckets from a joblib file."""
    if not path.exists():
        msg = f"Residual buckets file not found: {path}"
        raise FileNotFoundError(msg)
    result: dict[str, ResidualBuckets] = joblib.load(path)
    return result
