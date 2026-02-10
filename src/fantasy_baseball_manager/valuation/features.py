"""Feature extraction for ML valuation models.

Extracts numpy feature arrays from BatterTrainingRow / PitcherTrainingRow
(training time) and from BattingProjection / PitchingProjection (inference time).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fantasy_baseball_manager.adp.training_dataset import (
        BatterTrainingRow,
        PitcherTrainingRow,
    )
    from fantasy_baseball_manager.marcel.models import (
        BattingProjection,
        PitchingProjection,
    )

BATTER_FEATURE_NAMES: tuple[str, ...] = (
    "pa",
    "hr",
    "r",
    "rbi",
    "sb",
    "bb",
    "so",
    "obp",
    "slg",
    "war",
    "position",
)

PITCHER_FEATURE_NAMES: tuple[str, ...] = (
    "ip",
    "w",
    "nsvh",
    "gs",
    "so",
    "bb",
    "hr",
    "era",
    "whip",
)

# Position ordinal encodes scarcity: lower = scarcer.
_POSITION_ORDINAL: dict[str, int] = {
    "C": 1,
    "SS": 2,
    "2B": 3,
    "3B": 4,
    "OF": 5,
    "CF": 5,
    "LF": 5,
    "RF": 5,
    "1B": 6,
    "DH": 6,
}

_DEFAULT_POSITION_ORDINAL: int = 5


def position_to_ordinal(position: str) -> int:
    """Map position string to scarcity-based ordinal.

    C=1, SS=2, 2B=3, 3B=4, OF/CF/LF/RF=5, 1B/DH=6.
    Returns 5 (OF-level) for unknown positions.
    """
    return _POSITION_ORDINAL.get(position.strip().upper(), _DEFAULT_POSITION_ORDINAL)


def batter_training_rows_to_arrays(
    rows: list[BatterTrainingRow],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert batter training rows to feature and target arrays.

    Returns:
        Tuple of (X, y) where X has shape (n, 11) and y has shape (n,).
        y = log(adp) to compress the non-linear ADP tail.
    """
    n = len(rows)
    X = np.empty((n, len(BATTER_FEATURE_NAMES)), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)

    for i, row in enumerate(rows):
        X[i] = [
            row.pa,
            row.hr,
            row.r,
            row.rbi,
            row.sb,
            row.bb,
            row.so,
            row.obp,
            row.slg,
            row.war,
            position_to_ordinal(row.position),
        ]
        y[i] = math.log(row.adp)

    return X, y


def pitcher_training_rows_to_arrays(
    rows: list[PitcherTrainingRow],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pitcher training rows to feature and target arrays.

    Returns:
        Tuple of (X, y) where X has shape (n, 9) and y has shape (n,).
        y = log(adp) to compress the non-linear ADP tail.
    """
    n = len(rows)
    X = np.empty((n, len(PITCHER_FEATURE_NAMES)), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)

    for i, row in enumerate(rows):
        X[i] = [
            row.ip,
            row.w,
            row.sv + row.hld,
            row.gs,
            row.so,
            row.bb,
            row.hr,
            row.era,
            row.whip,
        ]
        y[i] = math.log(row.adp)

    return X, y


def batting_projection_to_features(
    proj: BattingProjection,
    position: str = "OF",
) -> np.ndarray:
    """Extract features from a BattingProjection for inference.

    BattingProjection (marcel.models) does not carry position, so the
    caller supplies it.  OBP and SLG are computed from counting stats.

    Returns:
        1-D array of shape (11,).
    """
    pa = float(proj.pa)
    h = float(proj.h)
    bb = float(proj.bb)
    hbp = float(proj.hbp)
    ab = float(proj.ab)

    obp = (h + bb + hbp) / pa if pa > 0 else 0.0
    slg = (
        (float(proj.singles) + 2 * float(proj.doubles) + 3 * float(proj.triples) + 4 * float(proj.hr)) / ab
        if ab > 0
        else 0.0
    )

    return np.array(
        [
            pa,
            float(proj.hr),
            float(proj.r),
            float(proj.rbi),
            float(proj.sb),
            float(proj.bb),
            float(proj.so),
            obp,
            slg,
            0.0,  # WAR not available on marcel BattingProjection
            position_to_ordinal(position),
        ],
        dtype=np.float64,
    )


def pitching_projection_to_features(proj: PitchingProjection) -> np.ndarray:
    """Extract features from a PitchingProjection for inference.

    Returns:
        1-D array of shape (9,).
    """
    return np.array(
        [
            float(proj.ip),
            float(proj.w),
            float(proj.nsvh),
            float(proj.gs),
            float(proj.so),
            float(proj.bb),
            float(proj.hr),
            float(proj.era),
            float(proj.whip),
        ],
        dtype=np.float64,
    )
