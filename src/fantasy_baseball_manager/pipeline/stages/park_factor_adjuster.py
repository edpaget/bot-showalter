from fantasy_baseball_manager.pipeline.park_factors import ParkFactorProvider
from fantasy_baseball_manager.pipeline.types import PlayerRates


class ParkFactorAdjuster:
    """Neutralizes park effects using half-game blending.

    Since players play roughly 50% of games at home and 50% on the
    road (neutral), the effective park factor is blended:

        adjusted_rate = raw_rate / (0.5 * park_factor + 0.5)

    Reads metadata["team"] to look up the corresponding park factor
    for each stat. Players with missing team metadata pass through
    unchanged.

    Park factors are lazily loaded on the first adjust() call using
    the projection year from the PlayerRates objects.
    """

    def __init__(self, provider: ParkFactorProvider) -> None:
        self._provider = provider
        self._factors: dict[str, dict[str, float]] | None = None
        self._cached_year: int | None = None

    def _get_factors(self, year: int) -> dict[str, dict[str, float]]:
        if self._factors is None or self._cached_year != year:
            self._factors = self._provider.park_factors(year)
            self._cached_year = year
        return self._factors

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        if not players:
            return []

        factors = self._get_factors(players[0].year)

        result: list[PlayerRates] = []
        for p in players:
            team = p.metadata.get("team", "")
            if not team or team not in factors:
                result.append(p)
                continue

            team_factors = factors[str(team)]
            new_rates: dict[str, float] = {}
            for stat, rate in p.rates.items():
                factor = team_factors.get(stat, 1.0)
                blended = 0.5 * factor + 0.5
                new_rates[stat] = rate / blended if blended != 0.0 else rate

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
