from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore

logger = logging.getLogger(__name__)


class ParkFactorProvider(Protocol):
    def park_factors(self, year: int) -> dict[str, dict[str, float]]:
        """Return park factors by team and stat.

        Returns a mapping of team abbreviation -> stat name -> factor,
        where factor > 1.0 means the park inflates that stat.
        """
        ...


# FanGraphs Guts page uses display names; pybaseball uses abbreviations.
_TEAM_NAME_TO_ABBREV: dict[str, str] = {
    "Angels": "LAA",
    "Orioles": "BAL",
    "Red Sox": "BOS",
    "White Sox": "CHW",
    "Guardians": "CLE",
    "Indians": "CLE",
    "Tigers": "DET",
    "Royals": "KCR",
    "Twins": "MIN",
    "Yankees": "NYY",
    "Athletics": "OAK",
    "Mariners": "SEA",
    "Rays": "TBR",
    "Rangers": "TEX",
    "Blue Jays": "TOR",
    "Diamondbacks": "ARI",
    "Braves": "ATL",
    "Cubs": "CHC",
    "Reds": "CIN",
    "Rockies": "COL",
    "Marlins": "MIA",
    "Astros": "HOU",
    "Dodgers": "LAD",
    "Brewers": "MIL",
    "Nationals": "WSN",
    "Mets": "NYM",
    "Phillies": "PHI",
    "Pirates": "PIT",
    "Cardinals": "STL",
    "Padres": "SDP",
    "Giants": "SFG",
}

_GUTS_URL = "https://www.fangraphs.com/guts.aspx?type=pf&teamid=0&season={year}"


class FanGraphsParkFactorProvider:
    """Provides park factors scraped from the FanGraphs Guts page.

    Averages multiple years of park factor data and regresses toward 1.0
    to reduce noise from single-year samples.
    """

    def __init__(
        self,
        *,
        years_to_average: int = 3,
        regression_weight: float = 0.5,
    ) -> None:
        self._years_to_average = years_to_average
        self._regression_weight = regression_weight

    def park_factors(self, year: int) -> dict[str, dict[str, float]]:
        import pandas as pd

        years = [year - i for i in range(self._years_to_average)]
        raw_factors: dict[str, list[dict[str, float]]] = {}

        for y in years:
            url = _GUTS_URL.format(year=y)
            tables = pd.read_html(url)
            # The park factors table has columns: Season, Team, Basic (5yr), ...
            df = _find_park_factors_table(tables)
            if df is None:
                continue

            for _, row in df.iterrows():
                display_name = str(row.get("Team", "")).strip()
                abbrev = _TEAM_NAME_TO_ABBREV.get(display_name, "")
                if not abbrev:
                    continue
                if abbrev not in raw_factors:
                    raw_factors[abbrev] = []
                factors: dict[str, float] = {}
                for col, stat in self._column_map().items():
                    val = row.get(col)
                    if val is not None:
                        factors[stat] = float(val) / 100.0
                raw_factors[abbrev].append(factors)

        result: dict[str, dict[str, float]] = {}
        for team, factor_list in raw_factors.items():
            averaged: dict[str, float] = {}
            all_stats = {s for f in factor_list for s in f}
            for stat in all_stats:
                vals = [f[stat] for f in factor_list if stat in f]
                raw_avg = sum(vals) / len(vals)
                averaged[stat] = self._regress(raw_avg)
            result[team] = averaged

        return result

    def _regress(self, raw_factor: float) -> float:
        """Regress a park factor toward 1.0."""
        w = self._regression_weight
        return w * raw_factor + (1.0 - w) * 1.0

    @staticmethod
    def _column_map() -> dict[str, str]:
        """Map FanGraphs park factor column names to stat names."""
        return {
            "HR": "hr",
            "1B": "singles",
            "2B": "doubles",
            "3B": "triples",
            "BB": "bb",
            "SO": "so",
        }


def _find_park_factors_table(tables: list) -> "object | None":
    """Find the park factors DataFrame among scraped HTML tables."""
    for table in tables:
        cols = set(table.columns)
        if "Team" in cols and "HR" in cols and "1B" in cols:
            return table
    return None


_NAMESPACE = "park_factors"
_DEFAULT_TTL = 60 * 60 * 24 * 30  # 30 days


class CachedParkFactorProvider:
    """Wraps a ParkFactorProvider with CacheStore-backed persistence."""

    def __init__(
        self,
        delegate: ParkFactorProvider,
        cache: CacheStore,
        ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        self._delegate = delegate
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    def park_factors(self, year: int) -> dict[str, dict[str, float]]:
        key = str(year)
        cached = self._cache.get(_NAMESPACE, key)
        if cached is not None:
            result: dict[str, dict[str, float]] = json.loads(cached)
            logger.debug("Cache hit for park_factors [year=%d] (%d teams)", year, len(result))
            return result
        logger.debug("Cache miss for park_factors [year=%d], fetching from source", year)
        result = self._delegate.park_factors(year)
        self._cache.put(_NAMESPACE, key, json.dumps(result), self._ttl_seconds)
        logger.debug("Cached park_factors [year=%d, ttl=%ds]", year, self._ttl_seconds)
        return result
