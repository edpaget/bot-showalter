"""ML-based valuation: projections → PlayerValue.

Bridges from BattingProjection/PitchingProjection to list[PlayerValue]
using a trained RidgeValuationModel.  This is the ML equivalent of
zscore_batting / zscore_pitching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fantasy_baseball_manager.valuation.features import (
    batting_projection_to_features,
    pitching_projection_to_features,
)
from fantasy_baseball_manager.valuation.models import PlayerValue

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.models import (
        BattingProjection,
        PitchingProjection,
    )
    from fantasy_baseball_manager.valuation.ridge_model import RidgeValuationModel


def ml_valuate_batting(
    projections: list[BattingProjection],
    model: RidgeValuationModel,
) -> list[PlayerValue]:
    """Apply learned valuation to batting projections.

    Returns list[PlayerValue] with:
    - total_value = negated predicted log(ADP) (higher = more valuable)
    - category_values = empty tuple (ML model predicts a single composite
      value, not per-category breakdowns)
    - position_type = "B"
    """
    if not projections:
        return []

    X = np.array(
        [batting_projection_to_features(p) for p in projections],
        dtype=np.float64,
    )
    values = model.predict_value(X)

    return [
        PlayerValue(
            player_id=proj.player_id,
            name=proj.name,
            category_values=(),
            total_value=float(val),
            position_type="B",
        )
        for proj, val in zip(projections, values, strict=True)
    ]


def ml_valuate_pitching(
    projections: list[PitchingProjection],
    model: RidgeValuationModel,
) -> list[PlayerValue]:
    """Apply learned valuation to pitching projections.

    Returns list[PlayerValue] with:
    - total_value = negated predicted log(ADP) (higher = more valuable)
    - category_values = empty tuple
    - position_type = "P"
    """
    if not projections:
        return []

    X = np.array(
        [pitching_projection_to_features(p) for p in projections],
        dtype=np.float64,
    )
    values = model.predict_value(X)

    return [
        PlayerValue(
            player_id=proj.player_id,
            name=proj.name,
            category_values=(),
            total_value=float(val),
            position_type="P",
        )
        for proj, val in zip(projections, values, strict=True)
    ]
