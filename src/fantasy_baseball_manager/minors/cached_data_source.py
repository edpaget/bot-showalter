"""Cached wrapper for minor league data sources."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, TypedDict

from fantasy_baseball_manager.minors.types import (
    MinorLeagueBatterSeasonStats,
    MinorLeagueLevel,
    MinorLeaguePitcherSeasonStats,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore
    from fantasy_baseball_manager.minors.data_source import MinorLeagueDataSource

logger = logging.getLogger(__name__)

_NAMESPACE = "milb_stats"
_TTL_HISTORICAL = 60 * 60 * 24 * 365  # 1 year for historical data
_TTL_CURRENT = 60 * 60 * 24  # 1 day for current season


class _BatterStatsDict(TypedDict):
    """TypedDict for serialized batter stats."""

    player_id: str
    name: str
    season: int
    age: int
    level: int
    team: str
    league: str
    pa: int
    ab: int
    h: int
    singles: int
    doubles: int
    triples: int
    hr: int
    rbi: int
    r: int
    bb: int
    so: int
    hbp: int
    sf: int
    sb: int
    cs: int
    avg: float
    obp: float
    slg: float


class _PitcherStatsDict(TypedDict):
    """TypedDict for serialized pitcher stats."""

    player_id: str
    name: str
    season: int
    age: int
    level: int
    team: str
    league: str
    g: int
    gs: int
    ip: float
    w: int
    losses: int
    sv: int
    h: int
    r: int
    er: int
    hr: int
    bb: int
    so: int
    hbp: int
    era: float
    whip: float


class CachedMinorLeagueDataSource:
    """Wraps MinorLeagueDataSource with CacheStore-backed persistence.

    Cache TTL strategy:
    - Historical seasons (year < current): 1 year TTL (data won't change)
    - Current season: 1 day TTL (updates during season)
    """

    def __init__(
        self,
        delegate: MinorLeagueDataSource,
        cache: CacheStore,
        current_year: int | None = None,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._current_year = current_year or datetime.now().year

    def batting_stats(self, year: int, level: MinorLeagueLevel) -> list[MinorLeagueBatterSeasonStats]:
        """Fetch batting stats with caching."""
        key = f"batting:{year}:{level.value}"
        cached = self._cache.get(_NAMESPACE, key)
        if cached is not None:
            result = _deserialize_batting(cached)
            logger.debug(
                "Cache hit for %s batting %d (%d players)",
                level.display_name,
                year,
                len(result),
            )
            return result

        logger.debug(
            "Cache miss for %s batting %d, fetching from source",
            level.display_name,
            year,
        )
        result = self._delegate.batting_stats(year, level)
        ttl = _TTL_CURRENT if year >= self._current_year else _TTL_HISTORICAL
        self._cache.put(_NAMESPACE, key, _serialize_batting(result), ttl)
        logger.debug(
            "Cached %d batters for %s %d [ttl=%ds]",
            len(result),
            level.display_name,
            year,
            ttl,
        )
        return result

    def batting_stats_all_levels(self, year: int) -> list[MinorLeagueBatterSeasonStats]:
        """Fetch batting stats across all levels with caching."""
        key = f"batting_all:{year}"
        cached = self._cache.get(_NAMESPACE, key)
        if cached is not None:
            result = _deserialize_batting(cached)
            logger.debug("Cache hit for all-level batting %d (%d players)", year, len(result))
            return result

        logger.debug("Cache miss for all-level batting %d, fetching from source", year)
        result = self._delegate.batting_stats_all_levels(year)
        ttl = _TTL_CURRENT if year >= self._current_year else _TTL_HISTORICAL
        self._cache.put(_NAMESPACE, key, _serialize_batting(result), ttl)
        logger.debug("Cached %d batters for all levels %d [ttl=%ds]", len(result), year, ttl)
        return result

    def pitching_stats(self, year: int, level: MinorLeagueLevel) -> list[MinorLeaguePitcherSeasonStats]:
        """Fetch pitching stats with caching."""
        key = f"pitching:{year}:{level.value}"
        cached = self._cache.get(_NAMESPACE, key)
        if cached is not None:
            result = _deserialize_pitching(cached)
            logger.debug(
                "Cache hit for %s pitching %d (%d players)",
                level.display_name,
                year,
                len(result),
            )
            return result

        logger.debug(
            "Cache miss for %s pitching %d, fetching from source",
            level.display_name,
            year,
        )
        result = self._delegate.pitching_stats(year, level)
        ttl = _TTL_CURRENT if year >= self._current_year else _TTL_HISTORICAL
        self._cache.put(_NAMESPACE, key, _serialize_pitching(result), ttl)
        logger.debug(
            "Cached %d pitchers for %s %d [ttl=%ds]",
            len(result),
            level.display_name,
            year,
            ttl,
        )
        return result

    def pitching_stats_all_levels(self, year: int) -> list[MinorLeaguePitcherSeasonStats]:
        """Fetch pitching stats across all levels with caching."""
        key = f"pitching_all:{year}"
        cached = self._cache.get(_NAMESPACE, key)
        if cached is not None:
            result = _deserialize_pitching(cached)
            logger.debug("Cache hit for all-level pitching %d (%d players)", year, len(result))
            return result

        logger.debug("Cache miss for all-level pitching %d, fetching from source", year)
        result = self._delegate.pitching_stats_all_levels(year)
        ttl = _TTL_CURRENT if year >= self._current_year else _TTL_HISTORICAL
        self._cache.put(_NAMESPACE, key, _serialize_pitching(result), ttl)
        logger.debug("Cached %d pitchers for all levels %d [ttl=%ds]", len(result), year, ttl)
        return result


def _serialize_batting(stats: list[MinorLeagueBatterSeasonStats]) -> str:
    """Serialize batter stats to JSON."""
    data: list[_BatterStatsDict] = []
    for s in stats:
        d = asdict(s)
        # Convert enum to int for serialization
        d["level"] = s.level.value
        data.append(d)  # type: ignore[arg-type]
    return json.dumps(data)


def _deserialize_batting(data: str) -> list[MinorLeagueBatterSeasonStats]:
    """Deserialize batter stats from JSON."""
    raw: list[_BatterStatsDict] = json.loads(data)
    return [
        MinorLeagueBatterSeasonStats(
            player_id=row["player_id"],
            name=row["name"],
            season=row["season"],
            age=row["age"],
            level=MinorLeagueLevel.from_sport_id(row["level"]),
            team=row["team"],
            league=row["league"],
            pa=row["pa"],
            ab=row["ab"],
            h=row["h"],
            singles=row["singles"],
            doubles=row["doubles"],
            triples=row["triples"],
            hr=row["hr"],
            rbi=row["rbi"],
            r=row["r"],
            bb=row["bb"],
            so=row["so"],
            hbp=row["hbp"],
            sf=row["sf"],
            sb=row["sb"],
            cs=row["cs"],
            avg=row["avg"],
            obp=row["obp"],
            slg=row["slg"],
        )
        for row in raw
    ]


def _serialize_pitching(stats: list[MinorLeaguePitcherSeasonStats]) -> str:
    """Serialize pitcher stats to JSON."""
    data: list[_PitcherStatsDict] = []
    for s in stats:
        d = asdict(s)
        # Convert enum to int for serialization
        d["level"] = s.level.value
        data.append(d)  # type: ignore[arg-type]
    return json.dumps(data)


def _deserialize_pitching(data: str) -> list[MinorLeaguePitcherSeasonStats]:
    """Deserialize pitcher stats from JSON."""
    raw: list[_PitcherStatsDict] = json.loads(data)
    return [
        MinorLeaguePitcherSeasonStats(
            player_id=row["player_id"],
            name=row["name"],
            season=row["season"],
            age=row["age"],
            level=MinorLeagueLevel.from_sport_id(row["level"]),
            team=row["team"],
            league=row["league"],
            g=row["g"],
            gs=row["gs"],
            ip=row["ip"],
            w=row["w"],
            losses=row["losses"],
            sv=row["sv"],
            h=row["h"],
            r=row["r"],
            er=row["er"],
            hr=row["hr"],
            bb=row["bb"],
            so=row["so"],
            hbp=row["hbp"],
            era=row["era"],
            whip=row["whip"],
        )
        for row in raw
    ]
