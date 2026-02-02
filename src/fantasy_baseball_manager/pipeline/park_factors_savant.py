from __future__ import annotations

import json
import logging
import re
import urllib.request

from fantasy_baseball_manager.pipeline.park_factors import _TEAM_NAME_TO_ABBREV

logger = logging.getLogger(__name__)

_SAVANT_URL = (
    "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
    "?type=year&year={year}&batSide={bat_side}"
    "&stat=index_wOBA&condition=All&rolling={rolling}"
)

_DATA_RE = re.compile(r"var\s+data\s*=\s*(\[.*?\])\s*;", re.DOTALL)

_STAT_MAP: dict[str, str] = {
    "index_hr": "hr",
    "index_1b": "singles",
    "index_2b": "doubles",
    "index_3b": "triples",
    "index_bb": "bb",
    "index_so": "so",
    "index_runs": "runs",
    "index_woba": "woba",
    "index_obp": "obp",
    "index_hits": "hits",
    "index_wobacon": "wobacon",
    "index_xwobacon": "xwobacon",
}


class SavantParkFactorProvider:
    """Provides park factors scraped from Baseball Savant's Statcast leaderboard.

    Savant factors are derived from home/away splits controlled for batted ball
    quality.  They are published on a 100-scale (100 = neutral) and can be
    filtered by batter handedness and rolling window.
    """

    def __init__(
        self,
        *,
        rolling_years: int = 3,
        bat_side: str = "",
    ) -> None:
        if rolling_years not in (1, 3):
            raise ValueError("rolling_years must be 1 or 3")
        if bat_side not in ("", "L", "R"):
            raise ValueError("bat_side must be '', 'L', or 'R'")
        self._rolling_years = rolling_years
        self._bat_side = bat_side

    def park_factors(self, year: int) -> dict[str, dict[str, float]]:
        html = self._fetch(year)
        entries = self._parse_json(html)
        return self._build_factors(entries)

    def _fetch(self, year: int) -> str:
        url = _SAVANT_URL.format(
            year=year,
            bat_side=self._bat_side,
            rolling=self._rolling_years,
        )
        logger.debug("Fetching Savant park factors: %s", url)
        with urllib.request.urlopen(url) as resp:  # noqa: S310
            return resp.read().decode("utf-8")

    @staticmethod
    def _parse_json(html: str) -> list[dict[str, str]]:
        match = _DATA_RE.search(html)
        if not match:
            logger.warning("Could not find park factor data in Savant HTML")
            return []
        return json.loads(match.group(1))  # type: ignore[no-any-return]

    @staticmethod
    def _build_factors(
        entries: list[dict[str, str]],
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for entry in entries:
            display_name = entry.get("name_display_club", "").strip()
            abbrev = _TEAM_NAME_TO_ABBREV.get(display_name, "")
            if not abbrev:
                logger.debug("Skipping unknown team: %s", display_name)
                continue
            factors: dict[str, float] = {}
            for savant_key, stat_name in _STAT_MAP.items():
                raw = entry.get(savant_key)
                if raw is not None:
                    factors[stat_name] = float(raw) / 100.0
            result[abbrev] = factors
        return result
