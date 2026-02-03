"""Split stats data source for platoon projections.

Provides batting stats separated by pitcher handedness (vs-LHP and vs-RHP).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import pybaseball
import pybaseball.cache
from pybaseball.datasources.fangraphs import FangraphsBattingStatsTable

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


class SplitStatsDataSource(Protocol):
    """Protocol for data sources that provide platoon split batting stats."""

    def batting_stats_vs_lhp(self, year: int) -> list[BattingSeasonStats]: ...
    def batting_stats_vs_rhp(self, year: int) -> list[BattingSeasonStats]: ...


def _fg_split_batting(year: int, split_month: int) -> list[BattingSeasonStats]:
    """Fetch batting stats for a specific FanGraphs split.

    Uses the FanGraphs leaderboard with a custom ``month`` value to select
    the platoon split (13 = vs LHP, 14 = vs RHP).
    """
    table = FangraphsBattingStatsTable()
    # Build query params manually so we can pass a split value that isn't
    # in pybaseball's FangraphsMonth enum.
    from pybaseball.enums.fangraphs import (
        FangraphsLeague,
        FangraphsPositions,
    )
    from pybaseball.enums.fangraphs.batting_data_enum import FangraphsBattingStats

    stat_columns = [col.value for col in FangraphsBattingStats]

    url_options = {
        "pos": FangraphsPositions.ALL.value,
        "stats": table.STATS_CATEGORY.value,
        "lg": FangraphsLeague.ALL.value,
        "qual": 0,
        "type": ",".join(str(v) for v in stat_columns),
        "season": year,
        "month": split_month,
        "season1": year,
        "ind": 1,
        "team": 0,
        "rost": 0,
        "age": "0,100",
        "filter": "",
        "players": "",
        "page": "1_1000000",
    }

    df = table._validate(
        table._postprocess(
            table.html_accessor.get_tabular_data_from_options(
                table.QUERY_ENDPOINT,
                query_params=url_options,
                column_name_mapper=table.COLUMN_NAME_MAPPER,  # type: ignore[arg-type]
                known_percentages=table.KNOWN_PERCENTAGES,
                row_id_func=table.ROW_ID_FUNC,
            )
        )
    )

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


class PybaseballSplitDataSource:
    """Fetches platoon split batting stats from FanGraphs via pybaseball."""

    def __init__(self) -> None:
        pybaseball.cache.enable()

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
