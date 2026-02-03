from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PitcherBattedBallStats:
    player_id: str  # FanGraphs ID
    name: str
    year: int
    pa: int  # batters faced (TBF)
    gb_pct: float
    fb_pct: float
    ld_pct: float
    iffb_pct: float


class PitcherBattedBallDataSource(Protocol):
    def pitcher_batted_ball_stats(self, year: int) -> list[PitcherBattedBallStats]: ...


class PybaseballBattedBallDataSource:
    """Fetches pitcher batted-ball profiles from FanGraphs via pybaseball."""

    def pitcher_batted_ball_stats(self, year: int) -> list[PitcherBattedBallStats]:
        from pybaseball import pitching_stats

        df = pitching_stats(year, qual=0)

        results: list[PitcherBattedBallStats] = []
        for _, row in df.iterrows():
            try:
                player_id = str(int(row["IDfg"]))
                name = str(row.get("Name", ""))
                pa = int(row.get("TBF", 0))
                gb_pct = float(row.get("GB%", 0))
                fb_pct = float(row.get("FB%", 0))
                ld_pct = float(row.get("LD%", 0))
                iffb_pct = float(row.get("IFFB%", 0))
            except (KeyError, ValueError, TypeError):
                continue

            results.append(
                PitcherBattedBallStats(
                    player_id=player_id,
                    name=name,
                    year=year,
                    pa=pa,
                    gb_pct=gb_pct,
                    fb_pct=fb_pct,
                    ld_pct=ld_pct,
                    iffb_pct=iffb_pct,
                )
            )
        logger.debug("Loaded %d batted-ball records for %d", len(results), year)
        return results


class CachedBattedBallDataSource:
    """Wraps any PitcherBattedBallDataSource with CacheStore."""

    def __init__(
        self,
        delegate: PitcherBattedBallDataSource,
        cache: CacheStore,
        ttl: int = 30 * 86400,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._ttl = ttl

    def pitcher_batted_ball_stats(self, year: int) -> list[PitcherBattedBallStats]:
        cache_key = f"pitcher_batted_ball_{year}"
        cached = self._cache.get("batted_ball", cache_key)
        if cached is not None:
            logger.debug("Batted-ball cache hit for year %d", year)
            rows = json.loads(cached)
            return [PitcherBattedBallStats(**row) for row in rows]

        logger.debug("Batted-ball cache miss for year %d, fetching", year)
        results = self._delegate.pitcher_batted_ball_stats(year)
        self._cache.put(
            "batted_ball",
            cache_key,
            json.dumps([asdict(s) for s in results]),
            self._ttl,
        )
        return results
