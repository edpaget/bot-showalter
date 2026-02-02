from fantasy_baseball_manager.marcel.data_source import StatsDataSource
from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)


def batting_stats_to_projection(stats: BattingSeasonStats) -> BattingProjection:
    return BattingProjection(
        player_id=stats.player_id,
        name=stats.name,
        year=stats.year,
        age=stats.age,
        pa=float(stats.pa),
        ab=float(stats.ab),
        h=float(stats.h),
        singles=float(stats.singles),
        doubles=float(stats.doubles),
        triples=float(stats.triples),
        hr=float(stats.hr),
        bb=float(stats.bb),
        so=float(stats.so),
        hbp=float(stats.hbp),
        sf=float(stats.sf),
        sh=float(stats.sh),
        sb=float(stats.sb),
        cs=float(stats.cs),
        r=float(stats.r),
        rbi=float(stats.rbi),
    )


def pitching_stats_to_projection(stats: PitchingSeasonStats) -> PitchingProjection:
    ip = stats.ip
    era = (stats.er / ip * 9) if ip > 0 else 0.0
    whip = ((stats.h + stats.bb) / ip) if ip > 0 else 0.0
    return PitchingProjection(
        player_id=stats.player_id,
        name=stats.name,
        year=stats.year,
        age=stats.age,
        ip=ip,
        g=float(stats.g),
        gs=float(stats.gs),
        er=float(stats.er),
        h=float(stats.h),
        bb=float(stats.bb),
        so=float(stats.so),
        hr=float(stats.hr),
        hbp=float(stats.hbp),
        era=era,
        whip=whip,
    )


def actuals_as_projections(
    data_source: StatsDataSource,
    year: int,
    min_pa: int = 0,
    min_ip: float = 0.0,
) -> tuple[list[BattingProjection], list[PitchingProjection]]:
    batting = [batting_stats_to_projection(s) for s in data_source.batting_stats(year) if s.pa >= min_pa]
    pitching = [pitching_stats_to_projection(s) for s in data_source.pitching_stats(year) if s.ip >= min_ip]
    return batting, pitching
