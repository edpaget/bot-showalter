from fantasy_baseball_manager.marcel.age_adjustment import age_multiplier
from fantasy_baseball_manager.marcel.league_averages import rebaseline
from fantasy_baseball_manager.pipeline.types import PlayerRates


class RebaselineAdjuster:
    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        result: list[PlayerRates] = []
        for p in players:
            avg_league_rates = p.metadata.get("avg_league_rates")
            target_rates = p.metadata.get("target_rates")
            if avg_league_rates is None or target_rates is None:
                result.append(p)
                continue

            new_rates = rebaseline(
                p.rates,
                avg_league_rates,
                target_rates,
            )
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


class MarcelAgingAdjuster:
    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        result: list[PlayerRates] = []
        for p in players:
            mult = age_multiplier(p.age)
            new_rates = {stat: rate * mult for stat, rate in p.rates.items()}
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
