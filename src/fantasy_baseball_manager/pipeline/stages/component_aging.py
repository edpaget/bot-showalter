"""Component-specific aging adjustment.

Applies per-stat aging curves instead of a uniform multiplier.
Different stats peak at different ages (e.g., speed peaks early,
plate discipline peaks late), and catchers decline faster than
other positions.
"""

from fantasy_baseball_manager.pipeline.stages.aging_curves import (
    BATTING_AGING_CURVES,
    PITCHING_AGING_CURVES,
    PITCHING_INVERTED_STATS,
    POSITION_AGING_MODIFIERS,
    AgingCurveParams,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates


def component_age_multiplier(
    age: int,
    curve: AgingCurveParams,
    position_modifier: float = 1.0,
) -> float:
    if age < curve.peak_age:
        return 1.0 + (curve.peak_age - age) * curve.young_rate
    elif age > curve.peak_age:
        return 1.0 - (age - curve.peak_age) * curve.old_rate * position_modifier
    return 1.0


class ComponentAgingAdjuster:
    def __init__(
        self,
        batting_curves: dict[str, AgingCurveParams] | None = None,
        pitching_curves: dict[str, AgingCurveParams] | None = None,
        pitching_inverted: frozenset[str] | None = None,
        position_modifiers: dict[str, float] | None = None,
    ) -> None:
        self._batting_curves = batting_curves or BATTING_AGING_CURVES
        self._pitching_curves = pitching_curves or PITCHING_AGING_CURVES
        self._pitching_inverted = pitching_inverted or PITCHING_INVERTED_STATS
        self._position_modifiers = position_modifiers or POSITION_AGING_MODIFIERS

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        result: list[PlayerRates] = []
        for p in players:
            is_pitcher = bool(p.metadata.get("is_starter"))
            position = str(p.metadata.get("position", ""))
            pos_modifier = self._position_modifiers.get(position, 1.0)
            curves = self._pitching_curves if is_pitcher else self._batting_curves

            new_rates: dict[str, float] = {}
            for stat, rate in p.rates.items():
                curve = curves.get(stat)
                if curve is None:
                    new_rates[stat] = rate
                    continue

                mult = component_age_multiplier(p.age, curve, pos_modifier)

                if is_pitcher and stat in self._pitching_inverted:
                    mult = 1.0 / mult if mult != 0 else 1.0

                new_rates[stat] = rate * mult

            result.append(
                PlayerRates(
                    player_id=p.player_id,
                    name=p.name,
                    year=p.year,
                    age=p.age,
                    rates=new_rates,
                    opportunities=p.opportunities,
                    metadata=p.metadata,
                )
            )
        return result
