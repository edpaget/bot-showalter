"""Save/load playing-time coefficients via joblib."""

from pathlib import Path

import joblib

from fantasy_baseball_manager.models.playing_time.engine import PlayingTimeCoefficients


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
