from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.marcel.weights import projected_ip
from fantasy_baseball_manager.pipeline.types import PlayerRates


class StandardFinalizer:
    def finalize_batting(self, players: list[PlayerRates]) -> list[BattingProjection]:
        result: list[BattingProjection] = []
        for p in players:
            proj_pa = p.opportunities  # for batting, opportunities = PA
            projected_stats = {stat: rate * proj_pa for stat, rate in p.rates.items()}

            proj_h = (
                projected_stats.get("singles", 0)
                + projected_stats.get("doubles", 0)
                + projected_stats.get("triples", 0)
                + projected_stats.get("hr", 0)
            )
            proj_ab = (
                proj_pa
                - projected_stats.get("bb", 0)
                - projected_stats.get("hbp", 0)
                - projected_stats.get("sf", 0)
                - projected_stats.get("sh", 0)
            )

            result.append(
                BattingProjection(
                    player_id=p.player_id,
                    name=p.name,
                    year=p.year,
                    age=p.age,
                    pa=proj_pa,
                    ab=proj_ab,
                    h=proj_h,
                    singles=projected_stats.get("singles", 0),
                    doubles=projected_stats.get("doubles", 0),
                    triples=projected_stats.get("triples", 0),
                    hr=projected_stats.get("hr", 0),
                    bb=projected_stats.get("bb", 0),
                    so=projected_stats.get("so", 0),
                    hbp=projected_stats.get("hbp", 0),
                    sf=projected_stats.get("sf", 0),
                    sh=projected_stats.get("sh", 0),
                    sb=projected_stats.get("sb", 0),
                    cs=projected_stats.get("cs", 0),
                    r=projected_stats.get("r", 0),
                    rbi=projected_stats.get("rbi", 0),
                )
            )
        return result

    def finalize_pitching(self, players: list[PlayerRates]) -> list[PitchingProjection]:
        result: list[PitchingProjection] = []
        for p in players:
            proj_outs = p.opportunities  # for pitching, opportunities = outs
            projected_stats = {stat: rate * proj_outs for stat, rate in p.rates.items()}

            is_starter = p.metadata.get("is_starter", True)
            ip_per_year = p.metadata.get("ip_per_year")

            if ip_per_year is not None:
                # Handle single float value (convert to list for consistency)
                ip_list = [ip_per_year] if isinstance(ip_per_year, (int, float)) else ip_per_year
                proj_ip = projected_ip(
                    ip_y1=ip_list[0],
                    ip_y2=ip_list[1] if len(ip_list) > 1 else 0,
                    is_starter=is_starter,
                )
            else:
                proj_ip = proj_outs / 3

            proj_er = projected_stats.get("er", 0)
            proj_h = projected_stats.get("h", 0)
            proj_bb = projected_stats.get("bb", 0)
            era = (proj_er / proj_ip) * 9 if proj_ip > 0 else 0.0
            whip = (proj_h + proj_bb) / proj_ip if proj_ip > 0 else 0.0

            proj_sv = projected_stats.get("sv", 0)
            proj_hld = projected_stats.get("hld", 0)
            proj_bs = projected_stats.get("bs", 0)
            nsvh = proj_sv + proj_hld - proj_bs

            if is_starter:
                proj_gs = proj_ip / 6.0
                proj_g = proj_gs
            else:
                proj_gs = 0.0
                proj_g = proj_ip

            result.append(
                PitchingProjection(
                    player_id=p.player_id,
                    name=p.name,
                    year=p.year,
                    age=p.age,
                    ip=proj_ip,
                    g=proj_g,
                    gs=proj_gs,
                    er=proj_er,
                    h=proj_h,
                    bb=proj_bb,
                    so=projected_stats.get("so", 0),
                    hr=projected_stats.get("hr", 0),
                    hbp=projected_stats.get("hbp", 0),
                    era=era,
                    whip=whip,
                    w=projected_stats.get("w", 0),
                    nsvh=nsvh,
                )
            )
        return result
