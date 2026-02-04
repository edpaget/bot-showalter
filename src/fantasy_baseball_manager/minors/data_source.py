"""Data sources for fetching minor league statistics."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import requests

from fantasy_baseball_manager.minors.types import (
    MinorLeagueBatterSeasonStats,
    MinorLeagueLevel,
    MinorLeaguePitcherSeasonStats,
)

logger = logging.getLogger(__name__)


class MinorLeagueDataSource(Protocol):
    """Protocol for fetching minor league statistics."""

    def batting_stats(
        self, year: int, level: MinorLeagueLevel
    ) -> list[MinorLeagueBatterSeasonStats]:
        """Fetch batting stats for a specific level and year."""
        ...

    def batting_stats_all_levels(
        self, year: int
    ) -> list[MinorLeagueBatterSeasonStats]:
        """Fetch batting stats across all MiLB levels for a year."""
        ...

    def pitching_stats(
        self, year: int, level: MinorLeagueLevel
    ) -> list[MinorLeaguePitcherSeasonStats]:
        """Fetch pitching stats for a specific level and year."""
        ...

    def pitching_stats_all_levels(
        self, year: int
    ) -> list[MinorLeaguePitcherSeasonStats]:
        """Fetch pitching stats across all MiLB levels for a year."""
        ...


class MLBStatsAPIDataSource:
    """Fetches minor league stats from MLB Stats API.

    The MLB Stats API provides minor league data via the sportId parameter:
    - AAA: sportId=11
    - AA: sportId=12
    - High-A: sportId=13
    - Single-A: sportId=14
    - Rookie: sportId=16
    """

    BASE_URL = "https://statsapi.mlb.com/api/v1"
    DEFAULT_TIMEOUT = 30
    DEFAULT_LIMIT = 1000

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    def batting_stats(
        self, year: int, level: MinorLeagueLevel
    ) -> list[MinorLeagueBatterSeasonStats]:
        """Fetch batting stats for a specific level and year."""
        response = requests.get(
            f"{self.BASE_URL}/stats",
            params={
                "stats": "season",
                "season": year,
                "sportId": level.value,
                "group": "hitting",
                "limit": self.DEFAULT_LIMIT,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return self._parse_batting_response(response.json(), year, level)

    def batting_stats_all_levels(
        self, year: int
    ) -> list[MinorLeagueBatterSeasonStats]:
        """Fetch batting stats across all MiLB levels for a year."""
        all_stats: list[MinorLeagueBatterSeasonStats] = []
        for level in MinorLeagueLevel:
            try:
                stats = self.batting_stats(year, level)
                all_stats.extend(stats)
                logger.debug(
                    "Fetched %d batters for %s %d", len(stats), level.display_name, year
                )
            except requests.RequestException as e:
                logger.warning(
                    "Failed to fetch %s batting for %d: %s",
                    level.display_name,
                    year,
                    e,
                )
        return all_stats

    def pitching_stats(
        self, year: int, level: MinorLeagueLevel
    ) -> list[MinorLeaguePitcherSeasonStats]:
        """Fetch pitching stats for a specific level and year."""
        response = requests.get(
            f"{self.BASE_URL}/stats",
            params={
                "stats": "season",
                "season": year,
                "sportId": level.value,
                "group": "pitching",
                "limit": self.DEFAULT_LIMIT,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return self._parse_pitching_response(response.json(), year, level)

    def pitching_stats_all_levels(
        self, year: int
    ) -> list[MinorLeaguePitcherSeasonStats]:
        """Fetch pitching stats across all MiLB levels for a year."""
        all_stats: list[MinorLeaguePitcherSeasonStats] = []
        for level in MinorLeagueLevel:
            try:
                stats = self.pitching_stats(year, level)
                all_stats.extend(stats)
                logger.debug(
                    "Fetched %d pitchers for %s %d",
                    len(stats),
                    level.display_name,
                    year,
                )
            except requests.RequestException as e:
                logger.warning(
                    "Failed to fetch %s pitching for %d: %s",
                    level.display_name,
                    year,
                    e,
                )
        return all_stats

    def _parse_batting_response(
        self, data: dict[str, Any], year: int, level: MinorLeagueLevel
    ) -> list[MinorLeagueBatterSeasonStats]:
        """Parse batting stats from MLB Stats API response."""
        results: list[MinorLeagueBatterSeasonStats] = []

        stats_data = data.get("stats", [])
        if not stats_data:
            return results

        splits = stats_data[0].get("splits", [])
        for split in splits:
            player_info = split.get("player", {})
            stat = split.get("stat", {})
            team_info = split.get("team", {})
            league_info = split.get("league", {})

            player_id = str(player_info.get("id", ""))
            if not player_id:
                continue

            h = int(stat.get("hits", 0))
            doubles = int(stat.get("doubles", 0))
            triples = int(stat.get("triples", 0))
            hr = int(stat.get("homeRuns", 0))
            singles = h - doubles - triples - hr

            results.append(
                MinorLeagueBatterSeasonStats(
                    player_id=player_id,
                    name=player_info.get("fullName", ""),
                    season=year,
                    age=self._parse_age(player_info.get("currentAge")),
                    level=level,
                    team=team_info.get("name", ""),
                    league=league_info.get("name", ""),
                    pa=int(stat.get("plateAppearances", 0)),
                    ab=int(stat.get("atBats", 0)),
                    h=h,
                    singles=singles,
                    doubles=doubles,
                    triples=triples,
                    hr=hr,
                    rbi=int(stat.get("rbi", 0)),
                    r=int(stat.get("runs", 0)),
                    bb=int(stat.get("baseOnBalls", 0)),
                    so=int(stat.get("strikeOuts", 0)),
                    hbp=int(stat.get("hitByPitch", 0)),
                    sf=int(stat.get("sacFlies", 0)),
                    sb=int(stat.get("stolenBases", 0)),
                    cs=int(stat.get("caughtStealing", 0)),
                    avg=self._parse_float(stat.get("avg", ".000")),
                    obp=self._parse_float(stat.get("obp", ".000")),
                    slg=self._parse_float(stat.get("slg", ".000")),
                )
            )

        return results

    def _parse_pitching_response(
        self, data: dict[str, Any], year: int, level: MinorLeagueLevel
    ) -> list[MinorLeaguePitcherSeasonStats]:
        """Parse pitching stats from MLB Stats API response."""
        results: list[MinorLeaguePitcherSeasonStats] = []

        stats_data = data.get("stats", [])
        if not stats_data:
            return results

        splits = stats_data[0].get("splits", [])
        for split in splits:
            player_info = split.get("player", {})
            stat = split.get("stat", {})
            team_info = split.get("team", {})
            league_info = split.get("league", {})

            player_id = str(player_info.get("id", ""))
            if not player_id:
                continue

            results.append(
                MinorLeaguePitcherSeasonStats(
                    player_id=player_id,
                    name=player_info.get("fullName", ""),
                    season=year,
                    age=self._parse_age(player_info.get("currentAge")),
                    level=level,
                    team=team_info.get("name", ""),
                    league=league_info.get("name", ""),
                    g=int(stat.get("gamesPlayed", 0)),
                    gs=int(stat.get("gamesStarted", 0)),
                    ip=self._parse_ip(stat.get("inningsPitched", "0.0")),
                    w=int(stat.get("wins", 0)),
                    losses=int(stat.get("losses", 0)),
                    sv=int(stat.get("saves", 0)),
                    h=int(stat.get("hits", 0)),
                    r=int(stat.get("runs", 0)),
                    er=int(stat.get("earnedRuns", 0)),
                    hr=int(stat.get("homeRuns", 0)),
                    bb=int(stat.get("baseOnBalls", 0)),
                    so=int(stat.get("strikeOuts", 0)),
                    hbp=int(stat.get("hitByPitch", 0)),
                    era=self._parse_float(stat.get("era", "0.00")),
                    whip=self._parse_float(stat.get("whip", "0.00")),
                )
            )

        return results

    def _parse_age(self, age: int | str | None) -> int:
        """Parse age from API response."""
        if age is None:
            return 0
        if isinstance(age, int):
            return age
        try:
            return int(age)
        except (ValueError, TypeError):
            return 0

    def _parse_float(self, value: str | float | None) -> float:
        """Parse float from API response (handles string stats like '.300')."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _parse_ip(self, value: str | float | None) -> float:
        """Parse innings pitched, handling '6.1' format (6 and 1/3 innings)."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            # Handle '6.1' = 6.33, '6.2' = 6.67 format
            parts = str(value).split(".")
            if len(parts) == 2:
                whole = int(parts[0])
                fraction = int(parts[1]) / 3.0 if parts[1] else 0.0
                return whole + fraction
            return float(value)
        except (ValueError, TypeError):
            return 0.0
