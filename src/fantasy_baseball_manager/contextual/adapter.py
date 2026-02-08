"""Adapter for converting per-game model predictions to per-PA/per-out rates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.data.models import GameSequence

# Out-producing pa_event values with their out counts
_OUT_EVENTS: dict[str, int] = {
    "field_out": 1,
    "strikeout": 1,
    "strikeout_double_play": 2,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "force_out": 1,
    "fielders_choice_out": 1,
    "sac_fly": 1,
    "sac_fly_double_play": 2,
    "sac_bunt": 1,
    "sac_bunt_double_play": 2,
    "triple_play": 3,
    "other_out": 1,
}

_BATTER_UNCOVERED_STATS = ("hbp", "sf", "sh", "sb", "cs", "r", "rbi")
_PITCHER_UNCOVERED_STATS = ("hbp", "er", "w", "sv", "hld", "bs")


class PerGameToSeasonAdapter:
    """Converts per-game count predictions to per-PA or per-out rates.

    For batters: divides predicted counts by average PA/game.
    For pitchers: divides predicted counts by average outs/game.
    """

    def __init__(self, perspective: str) -> None:
        self._perspective = perspective

    def count_plate_appearances(self, game: GameSequence) -> int:
        """Count pitches with a non-None pa_event (one PA per event)."""
        return sum(1 for p in game.pitches if p.pa_event is not None)

    def count_outs(self, game: GameSequence) -> int:
        """Count outs from pa_event values."""
        total = 0
        for p in game.pitches:
            if p.pa_event is not None:
                total += _OUT_EVENTS.get(p.pa_event, 0)
        return total

    def game_denominator(self, game: GameSequence) -> float:
        """Return PA for batters, outs for pitchers."""
        if self._perspective == "batter":
            return float(self.count_plate_appearances(game))
        return float(self.count_outs(game))

    def predictions_to_rates(
        self,
        avg_predictions: dict[str, float],
        avg_denominator: float,
        marcel_rates: dict[str, float],
    ) -> dict[str, float] | None:
        """Convert average per-game predictions to rates.

        Returns None if avg_denominator is zero (triggers fallback).
        """
        if avg_denominator == 0.0:
            return None

        rates: dict[str, float] = {}

        if self._perspective == "batter":
            self._convert_batter_rates(avg_predictions, avg_denominator, marcel_rates, rates)
        else:
            self._convert_pitcher_rates(avg_predictions, avg_denominator, marcel_rates, rates)

        return rates

    def _convert_batter_rates(
        self,
        preds: dict[str, float],
        denom: float,
        marcel_rates: dict[str, float],
        rates: dict[str, float],
    ) -> None:
        # Derive singles from h - 2b - 3b - hr, clamped to 0
        singles_count = max(
            0.0,
            preds.get("h", 0.0) - preds.get("2b", 0.0) - preds.get("3b", 0.0) - preds.get("hr", 0.0),
        )

        rates["hr"] = preds.get("hr", 0.0) / denom
        rates["so"] = preds.get("so", 0.0) / denom
        rates["bb"] = preds.get("bb", 0.0) / denom
        rates["singles"] = singles_count / denom
        rates["doubles"] = preds.get("2b", 0.0) / denom
        rates["triples"] = preds.get("3b", 0.0) / denom

        # Uncovered stats from Marcel
        for stat in _BATTER_UNCOVERED_STATS:
            if stat in marcel_rates:
                rates[stat] = marcel_rates[stat]

    def _convert_pitcher_rates(
        self,
        preds: dict[str, float],
        denom: float,
        marcel_rates: dict[str, float],
        rates: dict[str, float],
    ) -> None:
        rates["so"] = preds.get("so", 0.0) / denom
        rates["h"] = preds.get("h", 0.0) / denom
        rates["bb"] = preds.get("bb", 0.0) / denom
        rates["hr"] = preds.get("hr", 0.0) / denom

        # Uncovered stats from Marcel
        for stat in _PITCHER_UNCOVERED_STATS:
            if stat in marcel_rates:
                rates[stat] = marcel_rates[stat]
