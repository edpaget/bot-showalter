"""Valuator implementations and registry."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from fantasy_baseball_manager.valuation.models import StatCategory, ValuationResult, Valuator

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
    from fantasy_baseball_manager.valuation.ridge_model import RidgeValuationModel


class ZScoreValuator:
    """Valuator that uses z-score method for player valuation."""

    def valuate_batting(
        self,
        projections: list[BattingProjection],
        categories: tuple[StatCategory, ...],
    ) -> ValuationResult:
        from fantasy_baseball_manager.valuation.zscore import zscore_batting

        values = zscore_batting(projections, categories)
        return ValuationResult(values=values, categories=categories, label="Z-score")

    def valuate_pitching(
        self,
        projections: list[PitchingProjection],
        categories: tuple[StatCategory, ...],
    ) -> ValuationResult:
        from fantasy_baseball_manager.valuation.zscore import zscore_pitching

        values = zscore_pitching(projections, categories)
        return ValuationResult(values=values, categories=categories, label="Z-score")


class RidgeValuator:
    """Valuator that uses ML ridge regression for player valuation."""

    @cached_property
    def _batter_model(self) -> RidgeValuationModel:
        from fantasy_baseball_manager.valuation.ridge_model import load_model

        return load_model("default", "batter")

    @cached_property
    def _pitcher_model(self) -> RidgeValuationModel:
        from fantasy_baseball_manager.valuation.ridge_model import load_model

        return load_model("default", "pitcher")

    def valuate_batting(
        self,
        projections: list[BattingProjection],
        categories: tuple[StatCategory, ...],
    ) -> ValuationResult:
        from fantasy_baseball_manager.valuation.ml_valuate import ml_valuate_batting

        values = ml_valuate_batting(projections, self._batter_model)
        return ValuationResult(values=values, categories=(), label="ML Ridge")

    def valuate_pitching(
        self,
        projections: list[PitchingProjection],
        categories: tuple[StatCategory, ...],
    ) -> ValuationResult:
        from fantasy_baseball_manager.valuation.ml_valuate import ml_valuate_pitching

        values = ml_valuate_pitching(projections, self._pitcher_model)
        return ValuationResult(values=values, categories=(), label="ML Ridge")


VALUATORS: dict[str, Callable[[], Valuator]] = {
    "zscore": ZScoreValuator,
    "ml-ridge": RidgeValuator,
}
