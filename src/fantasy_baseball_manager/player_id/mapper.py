from __future__ import annotations

import csv
import io
import logging
import urllib.request
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore

logger = logging.getLogger(__name__)

SFBB_CSV_URL = "https://www.smartfantasybaseball.com/PLAYERIDMAPCSV"

# Yahoo assigns synthetic IDs to two-way players.
# Map each synthetic ID to the real Yahoo ID so SFBB lookup works.
_YAHOO_SPLIT_OVERRIDES: dict[str, str] = {
    "1000001": "10835",  # Shohei Ohtani (Batter)
    "1000002": "10835",  # Shohei Ohtani (Pitcher)
}


class PlayerIdMapper(Protocol):
    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None: ...
    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None: ...


class SfbbMapper:
    """Maps player IDs between Yahoo and FanGraphs using the SFBB ID map."""

    def __init__(
        self,
        yahoo_to_fg: dict[str, str],
        fg_to_yahoo: dict[str, str],
    ) -> None:
        self._yahoo_to_fg = yahoo_to_fg
        self._fg_to_yahoo = fg_to_yahoo

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return self._yahoo_to_fg.get(yahoo_id)

    def fangraphs_to_yahoo(self, fangraphs_id: str) -> str | None:
        return self._fg_to_yahoo.get(fangraphs_id)

    @property
    def yahoo_to_fg_map(self) -> dict[str, str]:
        return dict(self._yahoo_to_fg)

    @property
    def fg_to_yahoo_map(self) -> dict[str, str]:
        return dict(self._fg_to_yahoo)


def _download_sfbb_csv(csv_url: str) -> str:
    """Download raw CSV text from the SFBB player ID map."""
    req = urllib.request.Request(csv_url, headers={"User-Agent": "fantasy-baseball-manager"})
    with urllib.request.urlopen(req) as response:
        return response.read().decode("utf-8")


def _parse_sfbb_csv(csv_text: str) -> SfbbMapper:
    """Parse SFBB CSV text into an SfbbMapper."""
    yahoo_to_fg: dict[str, str] = {}
    fg_to_yahoo: dict[str, str] = {}

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        yahoo_id = row.get("YAHOOID", "").strip()
        fg_id = row.get("IDFANGRAPHS", "").strip()
        if yahoo_id and fg_id:
            yahoo_to_fg[yahoo_id] = fg_id
            fg_to_yahoo[fg_id] = yahoo_id

    for synthetic_id, real_id in _YAHOO_SPLIT_OVERRIDES.items():
        if real_id in yahoo_to_fg:
            yahoo_to_fg[synthetic_id] = yahoo_to_fg[real_id]

    logger.debug("SFBB mapper: %d yahoo↔fangraphs mappings", len(yahoo_to_fg))
    return SfbbMapper(yahoo_to_fg, fg_to_yahoo)


def build_sfbb_mapper(csv_url: str = SFBB_CSV_URL) -> SfbbMapper:
    """Download and parse the SFBB player ID map CSV."""
    csv_text = _download_sfbb_csv(csv_url)
    return _parse_sfbb_csv(csv_text)


def build_cached_sfbb_mapper(
    cache: CacheStore,
    cache_key: str,
    ttl: int,
    csv_url: str = SFBB_CSV_URL,
) -> SfbbMapper:
    """Build an SfbbMapper, caching the raw CSV text in a CacheStore."""
    cached = cache.get("sfbb_csv", cache_key)
    if cached is not None:
        logger.debug("Cache hit for sfbb_csv [key=%s]", cache_key)
        return _parse_sfbb_csv(cached)

    logger.debug("Cache miss for sfbb_csv [key=%s], downloading", cache_key)
    csv_text = _download_sfbb_csv(csv_url)
    cache.put("sfbb_csv", cache_key, csv_text, ttl)
    return _parse_sfbb_csv(csv_text)
