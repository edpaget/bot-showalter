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
                    pa_y1=pa_per_year[0],
                    pa_y2=pa_per_year[1] if len(pa_per_year) > 1 else 0,
                )
            elif ip_per_year is not None:
                is_starter = p.metadata.get("is_starter", True)
                # Handle single float value (convert to list for consistency)
                ip_list = [ip_per_year] if isinstance(ip_per_year, (int, float)) else ip_per_year
                proj_ip = projected_ip(
                    ip_y1=ip_list[0],
                    ip_y2=ip_list[1] if len(ip_list) > 1 else 0,
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
                    player=p.player,
                )
            )
        return result
