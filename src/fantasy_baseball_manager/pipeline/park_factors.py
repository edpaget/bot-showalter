from typing import Protocol


class ParkFactorProvider(Protocol):
    def park_factors(self, year: int) -> dict[str, dict[str, float]]:
        """Return park factors by team and stat.

        Returns a mapping of team abbreviation -> stat name -> factor,
        where factor > 1.0 means the park inflates that stat.
        """
        ...


class FanGraphsParkFactorProvider:
    """Provides park factors derived from FanGraphs data via pybaseball.

    Averages multiple years of park factor data and regresses toward 1.0
    to reduce noise from single-year samples.
    """

    def __init__(
        self,
        *,
        years_to_average: int = 3,
        regression_weight: float = 0.5,
    ) -> None:
        self._years_to_average = years_to_average
        self._regression_weight = regression_weight

    def park_factors(self, year: int) -> dict[str, dict[str, float]]:
        import pybaseball

        years = [year - i for i in range(self._years_to_average)]
        raw_factors: dict[str, list[dict[str, float]]] = {}

        for y in years:
            df = pybaseball.park_factors(y)
            for _, row in df.iterrows():
                team = str(row.get("Team", ""))
                if not team:
                    continue
                if team not in raw_factors:
                    raw_factors[team] = []
                factors: dict[str, float] = {}
                for col, stat in self._column_map().items():
                    val = row.get(col)
                    if val is not None:
                        factors[stat] = float(val) / 100.0
                raw_factors[team].append(factors)

        result: dict[str, dict[str, float]] = {}
        for team, factor_list in raw_factors.items():
            averaged: dict[str, float] = {}
            all_stats = {s for f in factor_list for s in f}
            for stat in all_stats:
                vals = [f[stat] for f in factor_list if stat in f]
                raw_avg = sum(vals) / len(vals)
                averaged[stat] = self._regress(raw_avg)
            result[team] = averaged

        return result

    def _regress(self, raw_factor: float) -> float:
        """Regress a park factor toward 1.0."""
        w = self._regression_weight
        return w * raw_factor + (1.0 - w) * 1.0

    @staticmethod
    def _column_map() -> dict[str, str]:
        """Map FanGraphs park factor column names to stat names."""
        return {
            "HR": "hr",
            "1B": "singles",
            "2B": "doubles",
            "3B": "triples",
            "BB": "bb",
            "SO": "so",
            "R": "r",
        }
