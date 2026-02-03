"""Split stats data source for platoon projections.

Provides batting stats separated by pitcher handedness (vs-LHP and vs-RHP).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Protocol

import requests

from fantasy_baseball_manager.marcel.data_source import (
    _deserialize_batting,
    _serialize_batting,
)
from fantasy_baseball_manager.marcel.models import BattingSeasonStats

if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore


logger = logging.getLogger(__name__)

# FanGraphs "month" parameter values for platoon splits
_FG_SPLIT_VS_LHP = 13
_FG_SPLIT_VS_RHP = 14

# FanGraphs new API endpoint for leaderboard data
_FG_API_URL = "https://www.fangraphs.com/api/leaders/major-league/data"

_HTML_TAG_RE = re.compile(r"<[^>]+>")


class SplitStatsDataSource(Protocol):
    """Protocol for data sources that provide platoon split batting stats."""

    def batting_stats_vs_lhp(self, year: int) -> list[BattingSeasonStats]: ...
    def batting_stats_vs_rhp(self, year: int) -> list[BattingSeasonStats]: ...


def _strip_html(value: object) -> str:
    """Remove HTML tags from a value, returning plain text."""
    s = str(value)
    return _HTML_TAG_RE.sub("", s).strip()


def _fg_split_batting(year: int, split_month: int) -> list[BattingSeasonStats]:
    """Fetch batting stats for a specific FanGraphs split.

    Uses the FanGraphs JSON API with a custom ``month`` value to select
    the platoon split (13 = vs LHP, 14 = vs RHP).  Paginates to fetch
    all players.
    """
    params = {
        "pos": "all",
        "stats": "bat",
        "lg": "all",
        "qual": 0,
        "type": "0",
        "season": year,
        "month": split_month,
        "season1": year,
        "ind": 1,
        "team": "",
        "rost": "",
        "age": "",
        "filter": "",
        "players": "",
        "startdate": "",
        "enddate": "",
        "pageitems": 2000000000,
        "pagenum": 1,
    }
    resp = requests.get(_FG_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    all_rows: list[dict[str, object]] = body.get("data", [])

    results: list[BattingSeasonStats] = []
    for row in all_rows:
        h = int(row.get("H", 0))
        doubles = int(row.get("2B", 0))
        triples = int(row.get("3B", 0))
        hr = int(row.get("HR", 0))
        singles = h - doubles - triples - hr
        results.append(
            BattingSeasonStats(
                player_id=str(int(row["playerid"])),
                name=_strip_html(row.get("Name", "")),
                year=year,
                age=int(row.get("Age", 0)),
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
                team=_strip_html(row.get("Team", "")),
            )
        )
    return results


class PybaseballSplitDataSource:
    """Fetches platoon split batting stats from the FanGraphs API."""

    def batting_stats_vs_lhp(self, year: int) -> list[BattingSeasonStats]:
        return _fg_split_batting(year, _FG_SPLIT_VS_LHP)

    def batting_stats_vs_rhp(self, year: int) -> list[BattingSeasonStats]:
        return _fg_split_batting(year, _FG_SPLIT_VS_RHP)


_NAMESPACE = "split_stats_data"
_DEFAULT_TTL = 60 * 60 * 24 * 30  # 30 days


class CachedSplitDataSource:
    """Wraps a SplitStatsDataSource with CacheStore-backed persistence."""

    def __init__(
        self,
        delegate: SplitStatsDataSource,
        cache: CacheStore,
        ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    def batting_stats_vs_lhp(self, year: int) -> list[BattingSeasonStats]:
        return self._cached_call("batting_stats_vs_lhp", year, self._delegate.batting_stats_vs_lhp)

    def batting_stats_vs_rhp(self, year: int) -> list[BattingSeasonStats]:
        return self._cached_call("batting_stats_vs_rhp", year, self._delegate.batting_stats_vs_rhp)

    def _cached_call(
        self,
        method_name: str,
        year: int,
        fetch: object,
    ) -> list[BattingSeasonStats]:
        key = f"{method_name}:{year}"
        cached = self._cache.get(_NAMESPACE, key)
        if cached is not None:
            logger.debug("Cache hit for %s [year=%d]", method_name, year)
            return _deserialize_batting(cached)
        logger.debug("Cache miss for %s [year=%d], fetching from source", method_name, year)
        result = fetch(year)  # type: ignore[operator]
        self._cache.put(_NAMESPACE, key, _serialize_batting(result), self._ttl_seconds)
        return result
