from __future__ import annotations

import logging
from typing import TYPE_CHECKING, overload

import pybaseball
import pybaseball.cache

from fantasy_baseball_manager.context import get_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSource, DataSourceError
from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)
from fantasy_baseball_manager.result import Err, Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.player.identity import Player

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataSource classes
# ---------------------------------------------------------------------------


class BattingDataSource:
    """DataSource for batting stats using pybaseball.

    Implements the DataSource[BattingSeasonStats] protocol with proper overloads
    so the type checker knows the return type based on the query type.

    Year is read from the ambient Context.

    Usage:
        init_context(year=2024)
        batting_source = BattingDataSource()
        result = batting_source(ALL_PLAYERS)
        if result.is_ok():
            stats = result.unwrap()  # Type checker knows: Sequence[BattingSeasonStats]
    """

    def __init__(self) -> None:
        pybaseball.cache.enable()

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[BattingSeasonStats]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[BattingSeasonStats]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[BattingSeasonStats] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[BattingSeasonStats]] | Ok[BattingSeasonStats] | Err[DataSourceError]:
        # Only ALL_PLAYERS queries are supported - return Err for others
        if query is not ALL_PLAYERS:
            return Err(DataSourceError("Only ALL_PLAYERS queries supported"))

        ctx = get_context()
        year = ctx.year

        try:
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
            return Ok(results)
        except Exception as e:
            return Err(DataSourceError(f"Failed to fetch batting stats for {year}", e))


def create_batting_source() -> DataSource[BattingSeasonStats]:
    """Create a DataSource for batting stats.

    Returns a BattingDataSource instance that implements the full DataSource protocol.

    Usage:
        init_context(year=2024)
        batting_source = create_batting_source()
        result = batting_source(ALL_PLAYERS)
        if result.is_ok():
            stats = result.unwrap()  # Type checker knows: Sequence[BattingSeasonStats]
    """
    return BattingDataSource()


class PitchingDataSource:
    """DataSource for pitching stats using pybaseball.

    Implements the DataSource[PitchingSeasonStats] protocol with proper overloads
    so the type checker knows the return type based on the query type.

    Year is read from the ambient Context.
    """

    def __init__(self) -> None:
        pybaseball.cache.enable()

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[PitchingSeasonStats]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[PitchingSeasonStats]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[PitchingSeasonStats] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[PitchingSeasonStats]] | Ok[PitchingSeasonStats] | Err[DataSourceError]:
        if query is not ALL_PLAYERS:
            return Err(DataSourceError("Only ALL_PLAYERS queries supported"))

        ctx = get_context()
        year = ctx.year

        try:
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
            return Ok(results)
        except Exception as e:
            return Err(DataSourceError(f"Failed to fetch pitching stats for {year}", e))


def create_pitching_source() -> DataSource[PitchingSeasonStats]:
    """Create a DataSource for pitching stats.

    Returns a PitchingDataSource instance that implements the full DataSource protocol.
    """
    return PitchingDataSource()


class TeamBattingDataSource:
    """DataSource for team batting stats using pybaseball.

    Implements the DataSource[BattingSeasonStats] protocol with proper overloads
    so the type checker knows the return type based on the query type.

    Year is read from the ambient Context.
    """

    def __init__(self) -> None:
        pybaseball.cache.enable()

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[BattingSeasonStats]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[BattingSeasonStats]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[BattingSeasonStats] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[BattingSeasonStats]] | Ok[BattingSeasonStats] | Err[DataSourceError]:
        if query is not ALL_PLAYERS:
            return Err(DataSourceError("Only ALL_PLAYERS queries supported"))

        ctx = get_context()
        year = ctx.year

        try:
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
            return Ok(results)
        except Exception as e:
            return Err(DataSourceError(f"Failed to fetch team batting stats for {year}", e))


def create_team_batting_source() -> DataSource[BattingSeasonStats]:
    """Create a DataSource for team batting stats.

    Returns a TeamBattingDataSource instance that implements the full DataSource protocol.
    """
    return TeamBattingDataSource()


class TeamPitchingDataSource:
    """DataSource for team pitching stats using pybaseball.

    Implements the DataSource[PitchingSeasonStats] protocol with proper overloads
    so the type checker knows the return type based on the query type.

    Year is read from the ambient Context.
    """

    def __init__(self) -> None:
        pybaseball.cache.enable()

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[PitchingSeasonStats]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[PitchingSeasonStats]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[PitchingSeasonStats] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[PitchingSeasonStats]] | Ok[PitchingSeasonStats] | Err[DataSourceError]:
        if query is not ALL_PLAYERS:
            return Err(DataSourceError("Only ALL_PLAYERS queries supported"))

        ctx = get_context()
        year = ctx.year

        try:
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
            return Ok(results)
        except Exception as e:
            return Err(DataSourceError(f"Failed to fetch team pitching stats for {year}", e))


def create_team_pitching_source() -> DataSource[PitchingSeasonStats]:
    """Create a DataSource for team pitching stats.

    Returns a TeamPitchingDataSource instance that implements the full DataSource protocol.
    """
    return TeamPitchingDataSource()
