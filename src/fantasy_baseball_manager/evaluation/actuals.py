from __future__ import annotations

from typing import TYPE_CHECKING

from fantasy_baseball_manager.context import new_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS
from fantasy_baseball_manager.marcel.models import (
    BattingProjection,
    BattingSeasonStats,
    PitchingProjection,
    PitchingSeasonStats,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource


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
    nsvh = float(stats.sv + stats.hld - stats.bs)
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
        w=float(stats.w),
        nsvh=nsvh,
    )


def actuals_as_projections(
    batting_source: DataSource[BattingSeasonStats],
    pitching_source: DataSource[PitchingSeasonStats],
    year: int,
    min_pa: int = 0,
    min_ip: float = 0.0,
) -> tuple[list[BattingProjection], list[PitchingProjection]]:
    with new_context(year=year):
        batting_result = batting_source(ALL_PLAYERS)
        pitching_result = pitching_source(ALL_PLAYERS)

    all_batting = list(batting_result.unwrap()) if batting_result.is_ok() else []
    all_pitching = list(pitching_result.unwrap()) if pitching_result.is_ok() else []

    batting = [batting_stats_to_projection(s) for s in all_batting if s.pa >= min_pa]
    pitching = [pitching_stats_to_projection(s) for s in all_pitching if s.ip >= min_ip]
    return batting, pitching
