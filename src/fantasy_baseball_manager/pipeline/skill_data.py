"""Skill data models and sources for detecting year-over-year skill changes."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatterSkillStats:
    """Skill metrics for a batter in a single season."""

    player_id: str  # FanGraphs ID (IDfg)
    name: str
    year: int
    pa: int

    # Contact quality (Tier 1)
    barrel_rate: float  # Barrel%
    hard_hit_rate: float  # HardHit%
    exit_velo_avg: float  # EV
    exit_velo_max: float  # maxEV

    # Plate discipline (Tier 1)
    chase_rate: float  # O-Swing%
    whiff_rate: float  # SwStr%

    # Speed (Tier 2)
    sprint_speed: float | None  # ft/sec, None if not available


@dataclass(frozen=True)
class PitcherSkillStats:
    """Skill metrics for a pitcher in a single season."""

    player_id: str  # FanGraphs ID (IDfg)
    name: str
    year: int
    pa_against: int  # Batters faced (TBF)

    # Stuff (Tier 1)
    fastball_velo: float | None  # vFA (sc)
    whiff_rate: float  # SwStr%

    # Batted ball (Tier 2)
    gb_rate: float  # GB%
    barrel_rate_against: float | None  # Barrel% against


class SkillDataSource(Protocol):
    """Protocol for fetching player skill data."""

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]: ...
    def pitcher_skill_stats(self, year: int) -> list[PitcherSkillStats]: ...


class SprintSpeedSource(Protocol):
    """Protocol for fetching sprint speed data."""

    def sprint_speeds(self, year: int) -> dict[str, float]: ...


class FanGraphsSkillDataSource:
    """Fetches skill metrics from FanGraphs via pybaseball."""

    def __init__(self, min_pa: int = 50, min_ip: int = 20) -> None:
        self._min_pa = min_pa
        self._min_ip = min_ip

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]:
        from pybaseball import batting_stats

        df = batting_stats(year, qual=self._min_pa)

        results: list[BatterSkillStats] = []
        for _, row in df.iterrows():
            # Skip rows missing required fields
            if "IDfg" not in row or "Name" not in row:
                continue

            results.append(
                BatterSkillStats(
                    player_id=str(row["IDfg"]),
                    name=str(row["Name"]),
                    year=year,
                    pa=int(row.get("PA", 0)),
                    barrel_rate=float(row.get("Barrel%", 0) or 0),
                    hard_hit_rate=float(row.get("HardHit%", 0) or 0),
                    exit_velo_avg=float(row.get("EV", 0) or 0),
                    exit_velo_max=float(row.get("maxEV", 0) or 0),
                    chase_rate=float(row.get("O-Swing%", 0) or 0),
                    whiff_rate=float(row.get("SwStr%", 0) or 0),
                    sprint_speed=None,  # Will be merged by CompositeSkillDataSource
                )
            )

        logger.debug("Loaded %d FanGraphs batter skill records for %d", len(results), year)
        return results

    def pitcher_skill_stats(self, year: int) -> list[PitcherSkillStats]:
        from pybaseball import pitching_stats

        df = pitching_stats(year, qual=self._min_ip)

        results: list[PitcherSkillStats] = []
        for _, row in df.iterrows():
            # Skip rows missing required fields
            if "IDfg" not in row or "Name" not in row:
                continue

            # Handle optional fastball velocity
            velo = row.get("vFA (sc)")
            fastball_velo = float(velo) if velo is not None and velo != "" else None

            results.append(
                PitcherSkillStats(
                    player_id=str(row["IDfg"]),
                    name=str(row["Name"]),
                    year=year,
                    pa_against=int(row.get("TBF", 0)),
                    fastball_velo=fastball_velo,
                    whiff_rate=float(row.get("SwStr%", 0) or 0),
                    gb_rate=float(row.get("GB%", 0) or 0),
                    barrel_rate_against=None,  # Not available in FanGraphs pitching_stats
                )
            )

        logger.debug("Loaded %d FanGraphs pitcher skill records for %d", len(results), year)
        return results


class StatcastSprintSpeedSource:
    """Fetches sprint speed from Statcast."""

    def sprint_speeds(self, year: int) -> dict[str, float]:
        """Return mapping of MLBAM player_id -> sprint_speed (ft/sec)."""
        from pybaseball import statcast_sprint_speed

        df = statcast_sprint_speed(year)
        result: dict[str, float] = {}

        for _, row in df.iterrows():
            pid = row.get("player_id")
            speed = row.get("sprint_speed")
            if pid is not None and speed is not None:
                result[str(int(pid))] = float(speed)

        logger.debug("Loaded %d sprint speed records for %d", len(result), year)
        return result


class CompositeSkillDataSource:
    """Combines FanGraphs and Statcast data."""

    def __init__(
        self,
        fangraphs_source: SkillDataSource,
        sprint_source: SprintSpeedSource,
        id_mapper: PlayerIdMapper,
    ) -> None:
        self._fangraphs = fangraphs_source
        self._sprint = sprint_source
        self._mapper = id_mapper

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]:
        # Get FanGraphs batter stats
        batters = self._fangraphs.batter_skill_stats(year)

        # Get sprint speeds (keyed by MLBAM ID)
        sprint_speeds = self._sprint.sprint_speeds(year)

        # Merge sprint speed into batter stats
        results: list[BatterSkillStats] = []
        for batter in batters:
            mlbam_id = self._mapper.fangraphs_to_mlbam(batter.player_id)
            sprint_speed = sprint_speeds.get(mlbam_id) if mlbam_id else None

            # Create new instance with sprint speed
            results.append(
                BatterSkillStats(
                    player_id=batter.player_id,
                    name=batter.name,
                    year=batter.year,
                    pa=batter.pa,
                    barrel_rate=batter.barrel_rate,
                    hard_hit_rate=batter.hard_hit_rate,
                    exit_velo_avg=batter.exit_velo_avg,
                    exit_velo_max=batter.exit_velo_max,
                    chase_rate=batter.chase_rate,
                    whiff_rate=batter.whiff_rate,
                    sprint_speed=sprint_speed,
                )
            )

        return results

    def pitcher_skill_stats(self, year: int) -> list[PitcherSkillStats]:
        # Pitchers don't need sprint speed, just pass through
        return self._fangraphs.pitcher_skill_stats(year)


class CachedSkillDataSource:
    """Caches skill data using CacheStore."""

    def __init__(
        self,
        delegate: SkillDataSource,
        cache: CacheStore,
        ttl: int = 30 * 86400,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._ttl = ttl

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]:
        cache_key = f"skill_batter_{year}"
        cached = self._cache.get("skill_data", cache_key)
        if cached is not None:
            logger.debug("Skill data cache hit for batter year %d", year)
            rows = json.loads(cached)
            return [BatterSkillStats(**row) for row in rows]

        logger.debug("Skill data cache miss for batter year %d, fetching", year)
        results = self._delegate.batter_skill_stats(year)
        self._cache.put(
            "skill_data",
            cache_key,
            json.dumps([asdict(s) for s in results]),
            self._ttl,
        )
        return results

    def pitcher_skill_stats(self, year: int) -> list[PitcherSkillStats]:
        cache_key = f"skill_pitcher_{year}"
        cached = self._cache.get("skill_data", cache_key)
        if cached is not None:
            logger.debug("Skill data cache hit for pitcher year %d", year)
            rows = json.loads(cached)
            return [PitcherSkillStats(**row) for row in rows]

        logger.debug("Skill data cache miss for pitcher year %d, fetching", year)
        results = self._delegate.pitcher_skill_stats(year)
        self._cache.put(
            "skill_data",
            cache_key,
            json.dumps([asdict(s) for s in results]),
            self._ttl,
        )
        return results
