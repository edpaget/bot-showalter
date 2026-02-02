from typing import Protocol

import pybaseball
import pybaseball.cache

from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)


class StatsDataSource(Protocol):
    def batting_stats(self, year: int) -> list[BattingSeasonStats]: ...
    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]: ...
    def team_batting(self, year: int) -> list[BattingSeasonStats]: ...
    def team_pitching(self, year: int) -> list[PitchingSeasonStats]: ...


class PybaseballDataSource:
    """Fetches stats from pybaseball, converting DataFrames to typed dataclasses."""

    def __init__(self) -> None:
        pybaseball.cache.enable()

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        df = pybaseball.batting_stats(year, qual=0)
        results: list[BattingSeasonStats] = []
        for _, row in df.iterrows():
            h = int(row.get("H", 0))
            doubles = int(row.get("2B", 0))
            triples = int(row.get("3B", 0))
            hr = int(row.get("HR", 0))
            singles = h - doubles - triples - hr
            results.append(
                BattingSeasonStats(
                    player_id=str(row["IDfg"]),
                    name=str(row["Name"]),
                    year=year,
                    age=int(row["Age"]),
                    pa=int(row.get("PA", 0)),
                    ab=int(row.get("AB", 0)),
                    h=h,
                    singles=singles,
                    doubles=doubles,
                    triples=triples,
                    hr=hr,
                    bb=int(row.get("BB", 0)),
                    so=int(row.get("SO", 0)),
                    hbp=int(row.get("HBP", 0)),
                    sf=int(row.get("SF", 0)),
                    sh=int(row.get("SH", 0)),
                    sb=int(row.get("SB", 0)),
                    cs=int(row.get("CS", 0)),
                    r=int(row.get("R", 0)),
                    rbi=int(row.get("RBI", 0)),
                    team=str(row.get("Team", "")),
                )
            )
        return results

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        df = pybaseball.pitching_stats(year, qual=0)
        results: list[PitchingSeasonStats] = []
        for _, row in df.iterrows():
            results.append(
                PitchingSeasonStats(
                    player_id=str(row["IDfg"]),
                    name=str(row["Name"]),
                    year=year,
                    age=int(row["Age"]),
                    ip=float(row.get("IP", 0)),
                    g=int(row.get("G", 0)),
                    gs=int(row.get("GS", 0)),
                    er=int(row.get("ER", 0)),
                    h=int(row.get("H", 0)),
                    bb=int(row.get("BB", 0)),
                    so=int(row.get("SO", 0)),
                    hr=int(row.get("HR", 0)),
                    hbp=int(row.get("HBP", 0)),
                    w=int(row.get("W", 0)),
                    sv=int(row.get("SV", 0)),
                    hld=int(row.get("HLD", 0)),
                    bs=int(row.get("BS", 0)),
                    team=str(row.get("Team", "")),
                )
            )
        return results

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        df = pybaseball.team_batting(year)
        results: list[BattingSeasonStats] = []
        for _, row in df.iterrows():
            h = int(row.get("H", 0))
            doubles = int(row.get("2B", 0))
            triples = int(row.get("3B", 0))
            hr = int(row.get("HR", 0))
            singles = h - doubles - triples - hr
            results.append(
                BattingSeasonStats(
                    player_id=str(row.get("teamIDfg", row.get("Team", ""))),
                    name=str(row.get("Team", "")),
                    year=year,
                    age=0,
                    pa=int(row.get("PA", 0)),
                    ab=int(row.get("AB", 0)),
                    h=h,
                    singles=singles,
                    doubles=doubles,
                    triples=triples,
                    hr=hr,
                    bb=int(row.get("BB", 0)),
                    so=int(row.get("SO", 0)),
                    hbp=int(row.get("HBP", 0)),
                    sf=int(row.get("SF", 0)),
                    sh=int(row.get("SH", 0)),
                    sb=int(row.get("SB", 0)),
                    cs=int(row.get("CS", 0)),
                    r=int(row.get("R", 0)),
                    rbi=int(row.get("RBI", 0)),
                )
            )
        return results

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        df = pybaseball.team_pitching(year)
        results: list[PitchingSeasonStats] = []
        for _, row in df.iterrows():
            results.append(
                PitchingSeasonStats(
                    player_id=str(row.get("teamIDfg", row.get("Team", ""))),
                    name=str(row.get("Team", "")),
                    year=year,
                    age=0,
                    ip=float(row.get("IP", 0)),
                    g=int(row.get("G", 0)),
                    gs=int(row.get("GS", 0)),
                    er=int(row.get("ER", 0)),
                    h=int(row.get("H", 0)),
                    bb=int(row.get("BB", 0)),
                    so=int(row.get("SO", 0)),
                    hr=int(row.get("HR", 0)),
                    hbp=int(row.get("HBP", 0)),
                    w=int(row.get("W", 0)),
                    sv=int(row.get("SV", 0)),
                    hld=int(row.get("HLD", 0)),
                    bs=int(row.get("BS", 0)),
                )
            )
        return results
