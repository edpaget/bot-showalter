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


class StatcastDataSource(Protocol):
    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]: ...


class PybaseballStatcastDataSource:
    """Wraps pybaseball.statcast_batter_expected_stats()."""

    def batter_expected_stats(self, year: int) -> list[StatcastBatterStats]:
        from pybaseball import statcast_batter_expected_stats

        df = statcast_batter_expected_stats(year)
        results: list[StatcastBatterStats] = []
        for _, row in df.iterrows():
            results.append(
                StatcastBatterStats(
                    player_id=str(int(row["player_id"])),
                    name=str(row.get("player_name", row.get("last_name, first_name", ""))),
                    year=year,
                    pa=int(row["pa"]),
                    barrel_rate=float(row.get("brl_percent", row.get("barrel_batted_rate", 0))),
                    hard_hit_rate=float(row.get("hard_hit_percent", 0)),
                    xwoba=float(row.get("est_woba", row.get("xwoba", 0))),
                    xba=float(row.get("est_ba", row.get("xba", 0))),
                    xslg=float(row.get("est_slg", row.get("xslg", 0))),
                )
            )
        logger.debug("Loaded %d Statcast batter records for %d", len(results), year)
        return results


class CachedStatcastDataSource:
    """Wraps any StatcastDataSource with CacheStore."""

    def __init__(
        self,
        delegate: StatcastDataSource,
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
