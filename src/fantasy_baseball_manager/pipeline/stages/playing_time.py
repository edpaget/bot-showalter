from fantasy_baseball_manager.marcel.weights import projected_ip, projected_pa
from fantasy_baseball_manager.pipeline.types import PlayerRates


class MarcelPlayingTime:
    def project(self, players: list[PlayerRates]) -> list[PlayerRates]:
        result: list[PlayerRates] = []
        for p in players:
            pa_per_year = p.metadata.get("pa_per_year")
            ip_per_year = p.metadata.get("ip_per_year")

            if pa_per_year is not None:
                opps = projected_pa(
                    pa_y1=pa_per_year[0],  # type: ignore[index]
                    pa_y2=pa_per_year[1] if len(pa_per_year) > 1 else 0,  # type: ignore[arg-type]
                )
            elif ip_per_year is not None:
                is_starter: bool = p.metadata.get("is_starter", True)  # type: ignore[assignment]
                proj_ip = projected_ip(
                    ip_y1=ip_per_year[0],  # type: ignore[index]
                    ip_y2=ip_per_year[1] if len(ip_per_year) > 1 else 0,  # type: ignore[arg-type]
                    is_starter=is_starter,
                )
                opps = proj_ip * 3  # convert to outs for rate multiplication
            else:
                opps = p.opportunities

            result.append(
                PlayerRates(
                    player_id=p.player_id,
                    name=p.name,
                    year=p.year,
                    age=p.age,
                    rates=p.rates,
                    opportunities=opps,
                    metadata=p.metadata,
                )
            )
        return result
