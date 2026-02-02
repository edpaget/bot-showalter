from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Protocol

import pybaseball
import pybaseball.cache

from fantasy_baseball_manager.marcel.models import (
    BattingSeasonStats,
    PitchingSeasonStats,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from fantasy_baseball_manager.cache.protocol import CacheStore

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

_BATTING_FIELDS = (
    "player_id",
    "name",
    "year",
    "age",
    "pa",
    "ab",
    "h",
    "singles",
    "doubles",
    "triples",
    "hr",
    "bb",
    "so",
    "hbp",
    "sf",
    "sh",
    "sb",
    "cs",
    "r",
    "rbi",
    "team",
)

_PITCHING_FIELDS = (
    "player_id",
    "name",
    "year",
    "age",
    "ip",
    "g",
    "gs",
    "er",
    "h",
    "bb",
    "so",
    "hr",
    "hbp",
    "w",
    "sv",
    "hld",
    "bs",
    "team",
)


def _serialize_batting(stats: list[BattingSeasonStats]) -> str:
    return json.dumps([{f: getattr(s, f) for f in _BATTING_FIELDS} for s in stats])


def _deserialize_batting(data: str) -> list[BattingSeasonStats]:
    return [BattingSeasonStats(**row) for row in json.loads(data)]


def _serialize_pitching(stats: list[PitchingSeasonStats]) -> str:
    return json.dumps([{f: getattr(s, f) for f in _PITCHING_FIELDS} for s in stats])


def _deserialize_pitching(data: str) -> list[PitchingSeasonStats]:
    return [PitchingSeasonStats(**row) for row in json.loads(data)]


# ---------------------------------------------------------------------------
# CachedStatsDataSource
# ---------------------------------------------------------------------------

_NAMESPACE = "stats_data"
_DEFAULT_TTL = 60 * 60 * 24 * 30  # 30 days


class CachedStatsDataSource:
    """Wraps a StatsDataSource with CacheStore-backed persistence."""

    def __init__(
        self,
        delegate: StatsDataSource,
        cache: CacheStore,
        ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    def _cached_call(
        self,
        method_name: str,
        year: int,
        fetch: Callable[[int], Any],
        serialize: Callable[..., str],
        deserialize: Callable[[str], Any],
    ) -> Any:
        key = f"{method_name}:{year}"
        cached = self._cache.get(_NAMESPACE, key)
        if cached is not None:
            logger.debug("Cache hit for %s [year=%d]", method_name, year)
            return deserialize(cached)
        logger.debug("Cache miss for %s [year=%d], fetching from source", method_name, year)
        result = fetch(year)
        self._cache.put(_NAMESPACE, key, serialize(result), self._ttl_seconds)
        return result

    def batting_stats(self, year: int) -> list[BattingSeasonStats]:
        return self._cached_call(
            "batting_stats",
            year,
            self._delegate.batting_stats,
            _serialize_batting,
            _deserialize_batting,
        )

    def pitching_stats(self, year: int) -> list[PitchingSeasonStats]:
        return self._cached_call(
            "pitching_stats",
            year,
            self._delegate.pitching_stats,
            _serialize_pitching,
            _deserialize_pitching,
        )

    def team_batting(self, year: int) -> list[BattingSeasonStats]:
        return self._cached_call(
            "team_batting",
            year,
            self._delegate.team_batting,
            _serialize_batting,
            _deserialize_batting,
        )

    def team_pitching(self, year: int) -> list[PitchingSeasonStats]:
        return self._cached_call(
            "team_pitching",
            year,
            self._delegate.team_pitching,
            _serialize_pitching,
            _deserialize_pitching,
        )
