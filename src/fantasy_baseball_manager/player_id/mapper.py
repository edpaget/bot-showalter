from __future__ import annotations

import csv
import io
import logging
import urllib.request
from typing import TYPE_CHECKING, overload

from fantasy_baseball_manager.result import Err, Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.cache.protocol import CacheStore
    from fantasy_baseball_manager.player.identity import Player

logger = logging.getLogger(__name__)

SFBB_CSV_URL = "https://www.smartfantasybaseball.com/PLAYERIDMAPCSV"

# Yahoo assigns synthetic IDs to two-way players.
# Map each synthetic ID to the real Yahoo ID so SFBB lookup works.
_YAHOO_SPLIT_OVERRIDES: dict[str, str] = {
    "1000001": "10835",  # Shohei Ohtani (Batter)
    "1000002": "10835",  # Shohei Ohtani (Pitcher)
}


class PlayerMapperError(Exception):
    """Raised when player ID mapping fails.

    Attributes:
        message: Human-readable error description.
        cause: Optional underlying exception that caused this error.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message}: {self.cause}"
        return self.message


class SfbbMapper:
    """Maps player IDs between Yahoo, FanGraphs, and MLBAM using the SFBB ID map."""

    def __init__(
        self,
        yahoo_to_fg: dict[str, str],
        fg_to_yahoo: dict[str, str],
        fg_to_mlbam: dict[str, str] | None = None,
        mlbam_to_fg: dict[str, str] | None = None,
    ) -> None:
        self._yahoo_to_fg = yahoo_to_fg
        self._fg_to_yahoo = fg_to_yahoo
        self._fg_to_mlbam: dict[str, str] = fg_to_mlbam or {}
        self._mlbam_to_fg: dict[str, str] = mlbam_to_fg or {}

    def yahoo_to_fangraphs(self, yahoo_id: str) -> str | None:
        return self._yahoo_to_fg.get(yahoo_id)

    def fangraphs_to_mlbam(self, fangraphs_id: str) -> str | None:
        return self._fg_to_mlbam.get(fangraphs_id)

    def mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
        return self._mlbam_to_fg.get(mlbam_id)

    @property
    def yahoo_to_fg_map(self) -> dict[str, str]:
        return dict(self._yahoo_to_fg)

    # DataSource-style callable interface

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[Player]] | Err[PlayerMapperError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[Player] | Err[PlayerMapperError]: ...

    def __call__(self, query: Player | list[Player]) -> Ok[Player] | Ok[list[Player]] | Err[PlayerMapperError]:
        """Enrich Player(s) with additional IDs.

        Takes Player objects and returns new Player objects with fangraphs_id
        and mlbam_id fields populated based on the yahoo_id.

        Args:
            query: A single Player or list of Players to enrich.

        Returns:
            Ok containing the enriched Player(s), or Err if mapping fails.
            For unmapped IDs, the Player is returned unchanged (IDs remain None).
        """
        if isinstance(query, list):
            return Ok([self._enrich_player(p) for p in query])
        return Ok(self._enrich_player(query))

    def _enrich_player(self, player: Player) -> Player:
        """Enrich a single Player with IDs from the mapping tables."""
        # Use effective_yahoo_id which handles two-way player synthetic IDs
        yahoo_id = player.effective_yahoo_id

        fg_id = self._yahoo_to_fg.get(yahoo_id)
        mlbam_id = self._fg_to_mlbam.get(fg_id) if fg_id else None

        return player.with_ids(fangraphs_id=fg_id, mlbam_id=mlbam_id)


def _download_sfbb_csv(csv_url: str) -> str:
    """Download raw CSV text from the SFBB player ID map."""
    req = urllib.request.Request(csv_url, headers={"User-Agent": "fantasy-baseball-manager"})
    with urllib.request.urlopen(req) as response:
        return response.read().decode("utf-8")


def _parse_sfbb_csv(csv_text: str) -> SfbbMapper:
    """Parse SFBB CSV text into an SfbbMapper."""
    yahoo_to_fg: dict[str, str] = {}
    fg_to_yahoo: dict[str, str] = {}
    fg_to_mlbam: dict[str, str] = {}
    mlbam_to_fg: dict[str, str] = {}

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        yahoo_id = row.get("YAHOOID", "").strip()
        fg_id = row.get("IDFANGRAPHS", "").strip()
        mlbam_id = row.get("MLBID", "").strip()
        if yahoo_id and fg_id:
            yahoo_to_fg[yahoo_id] = fg_id
            fg_to_yahoo[fg_id] = yahoo_id
        if fg_id and mlbam_id:
            fg_to_mlbam[fg_id] = mlbam_id
            mlbam_to_fg[mlbam_id] = fg_id

    for synthetic_id, real_id in _YAHOO_SPLIT_OVERRIDES.items():
        if real_id in yahoo_to_fg:
            yahoo_to_fg[synthetic_id] = yahoo_to_fg[real_id]

    logger.debug(
        "SFBB mapper: %d yahoo<->fangraphs, %d fangraphs<->mlbam mappings",
        len(yahoo_to_fg),
        len(fg_to_mlbam),
    )
    return SfbbMapper(yahoo_to_fg, fg_to_yahoo, fg_to_mlbam, mlbam_to_fg)


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
