"""IP calibration via isotonic regression for pitcher playing-time predictions."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class IPCalibrator:
    """Calibrates pitcher IP predictions via isotonic regression.

    Stores breakpoints rather than the sklearn object to avoid pickling issues.
    """

    x_thresholds: tuple[float, ...]  # sorted raw prediction breakpoints
    y_calibrated: tuple[float, ...]  # corresponding calibrated values


def fit_ip_calibrator(
    predicted: list[float],
    actual: list[float],
) -> IPCalibrator:
    """Fit isotonic regression on (predicted, actual) pairs.

    Uses sklearn IsotonicRegression with out_of_bounds='clip'.
    Stores the fitted breakpoints for serialization.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(predicted, actual)

    x_thresholds = tuple(float(x) for x in iso.X_thresholds_)
    y_calibrated = tuple(float(y) for y in iso.y_thresholds_)

    return IPCalibrator(x_thresholds=x_thresholds, y_calibrated=y_calibrated)


def calibrate_ip(raw_ip: float, calibrator: IPCalibrator) -> float:
    """Map a single raw prediction through the calibrator.

    Uses numpy interp on the stored breakpoints.
    """
    return float(np.interp(raw_ip, calibrator.x_thresholds, calibrator.y_calibrated))


def calibrate_ip_batch(
    predictions: list[dict[str, Any]],
    calibrator: IPCalibrator | None,
    target_pitcher_count: int | None = None,
) -> list[dict[str, Any]]:
    """Post-process a batch of predictions.

    1. Apply isotonic calibration to all pitcher IP values.
    2. If target_pitcher_count given, keep only top N pitchers by
       calibrated IP; exclude the rest.
    3. Re-clamp calibrated values to [0, 250].
    4. Batter predictions pass through unchanged.
    Returns a new list (does not mutate input).
    """
    batters: list[dict[str, Any]] = []
    pitchers: list[dict[str, Any]] = []

    for pred in predictions:
        row = dict(pred)  # shallow copy
        if row.get("player_type") == "pitcher":
            if calibrator is not None:
                row["ip"] = calibrate_ip(float(row["ip"]), calibrator)
            row["ip"] = max(0.0, min(250.0, row["ip"]))
            pitchers.append(row)
        else:
            batters.append(row)

    if target_pitcher_count is not None:
        pitchers.sort(key=lambda p: p["ip"], reverse=True)
        pitchers = pitchers[:target_pitcher_count]

    return batters + pitchers
