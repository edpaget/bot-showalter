from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StatcastBatterStats:
    player_id: str  # MLBAM ID
    name: str
    year: int
    pa: int
    barrel_rate: float  # barrels / batted ball event
    hard_hit_rate: float
    xwoba: float
    xba: float
    xslg: float


@dataclass(frozen=True)
class StatcastPitcherStats:
    player_id: str  # MLBAM ID
    name: str
    year: int
    pa: int  # PA against
    xba: float  # expected BA against
    xslg: float  # expected SLG against
    xwoba: float  # expected wOBA against
    xera: float  # expected ERA
    barrel_rate: float  # barrel% against
    hard_hit_rate: float  # hard-hit% against


class StatcastDataSource(Protocol):
    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]: ...


class PitcherStatcastDataSource(Protocol):
    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]: ...


class PybaseballStatcastDataSource:
    """Merges pybaseball expected-stats and exit-velocity/barrel endpoints."""

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        from pybaseball import statcast_batter_exitvelo_barrels, statcast_batter_expected_stats

        xstats = statcast_batter_expected_stats(year)
        barrels = statcast_batter_exitvelo_barrels(year)

        barrel_lookup: dict[int, tuple[float, float]] = {}
        for _, row in barrels.iterrows():
            pid = int(row["player_id"])
            brl_pct = float(row.get("brl_percent", 0))
            hh_pct = float(row.get("ev95percent", 0))
            barrel_lookup[pid] = (brl_pct / 100.0, hh_pct / 100.0)

        results: list[StatcastBatterStats] = []
        for _, row in xstats.iterrows():
            pid = int(row["player_id"])
            barrel_rate, hard_hit_rate = barrel_lookup.get(pid, (0.0, 0.0))
            results.append(
                StatcastBatterStats(
                    player_id=str(pid),
                    name=str(row.get("player_name", row.get("last_name, first_name", ""))),
                    year=year,
                    pa=int(row["pa"]),
                    barrel_rate=barrel_rate,
                    hard_hit_rate=hard_hit_rate,
                    xwoba=float(row.get("est_woba", row.get("xwoba", 0))),
                    xba=float(row.get("est_ba", row.get("xba", 0))),
                    xslg=float(row.get("est_slg", row.get("xslg", 0))),
                )
            )
        logger.debug("Loaded %d Statcast batter records for %d", len(results), year)
        return results

    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]:
        from pybaseball import statcast_pitcher_exitvelo_barrels, statcast_pitcher_expected_stats

        xstats = statcast_pitcher_expected_stats(year, minPA=1)
        barrels = statcast_pitcher_exitvelo_barrels(year, minBBE=1)

        barrel_lookup: dict[int, tuple[float, float]] = {}
        for _, row in barrels.iterrows():
            pid = int(row["player_id"])
            brl_pct = float(row.get("brl_percent", 0))
            hh_pct = float(row.get("ev95percent", 0))
            barrel_lookup[pid] = (brl_pct / 100.0, hh_pct / 100.0)

        results: list[StatcastPitcherStats] = []
        for _, row in xstats.iterrows():
            pid = int(row["player_id"])
            barrel_rate, hard_hit_rate = barrel_lookup.get(pid, (0.0, 0.0))
            results.append(
                StatcastPitcherStats(
                    player_id=str(pid),
                    name=str(row.get("player_name", row.get("last_name, first_name", ""))),
                    year=year,
                    pa=int(row["pa"]),
                    xba=float(row.get("est_ba", row.get("xba", 0))),
                    xslg=float(row.get("est_slg", row.get("xslg", 0))),
                    xwoba=float(row.get("est_woba", row.get("xwoba", 0))),
                    xera=float(row.get("xera", 0)),
                    barrel_rate=barrel_rate,
                    hard_hit_rate=hard_hit_rate,
                )
            )
        logger.debug("Loaded %d Statcast pitcher records for %d", len(results), year)
        return results


class FullStatcastDataSource(Protocol):
    """Data source that provides both batter and pitcher Statcast data."""

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]: ...
    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]: ...


class CachedStatcastDataSource:
    """Wraps any FullStatcastDataSource with CacheStore."""

    def __init__(
        self,
        delegate: FullStatcastDataSource,
        cache: CacheStore,
        ttl: int = 30 * 86400,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._ttl = ttl

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        cache_key = f"statcast_batter_{year}"
        cached = self._cache.get("statcast", cache_key)
        if cached is not None:
            logger.debug("Statcast cache hit for year %d", year)
            rows = json.loads(cached)
            return [StatcastBatterStats(**row) for row in rows]

        logger.debug("Statcast cache miss for year %d, fetching", year)
        results = self._delegate.batter_expected_stats(year)
        self._cache.put(
            "statcast",
            cache_key,
            json.dumps([asdict(s) for s in results]),
            self._ttl,
        )
        return results

    def pitcher_expected_stats(self, year: int) -> list[StatcastPitcherStats]:
        cache_key = f"statcast_pitcher_{year}"
        cached = self._cache.get("statcast", cache_key)
        if cached is not None:
            logger.debug("Statcast pitcher cache hit for year %d", year)
            rows = json.loads(cached)
            return [StatcastPitcherStats(**row) for row in rows]

        logger.debug("Statcast pitcher cache miss for year %d, fetching", year)
        results = self._delegate.pitcher_expected_stats(year)
        self._cache.put(
            "statcast",
            cache_key,
            json.dumps([asdict(s) for s in results]),
            self._ttl,
        )
        return results
