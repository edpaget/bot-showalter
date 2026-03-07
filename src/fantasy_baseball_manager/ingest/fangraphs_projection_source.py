import logging
from typing import TYPE_CHECKING, Any

import httpx

from fantasy_baseball_manager.ingest._retry import default_http_retry

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.fangraphs.com/api/projections"

_DEFAULT_RETRY = default_http_retry("FanGraphs projections API call")

_VALID_PROJECTION_TYPES = frozenset(
    {
        "fangraphsdc",
        "steamer",
        "steamerr",
        "steameru",
        "zips",
        "rzips",
        "rfangraphsdc",
    }
)

_VALID_STAT_TYPES = frozenset({"bat", "pit"})

PROJECTION_SYSTEMS: dict[str, str] = {
    "fangraphs-dc": "fangraphsdc",
    "steamer": "steamer",
    "zips": "zips",
}


def _parse_response(response: httpx.Response) -> list[dict[str, Any]]:
    """Parse JSON response, handling both bare array and {"data": [...]} formats."""
    data = response.json()
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def _remap_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remap API field names so downstream projection mappers work unchanged.

    - ``playerid`` → ``PlayerId`` (keep original too)
    - ``xMLBAMID`` → ``MLBAMID``
    """
    for row in rows:
        if "playerid" in row:
            row["PlayerId"] = row["playerid"]
        if "xMLBAMID" in row:
            row["MLBAMID"] = row["xMLBAMID"]
    return rows


class FgProjectionSource:
    def __init__(
        self,
        projection_type: str,
        stat_type: str,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        if projection_type not in _VALID_PROJECTION_TYPES:
            raise ValueError(
                f"Unknown projection_type {projection_type!r}; expected one of {sorted(_VALID_PROJECTION_TYPES)}"
            )
        if stat_type not in _VALID_STAT_TYPES:
            raise ValueError(f"Unknown stat_type {stat_type!r}; expected one of {sorted(_VALID_STAT_TYPES)}")
        self._projection_type = projection_type
        self._stat_type = stat_type
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._fetch_with_retry = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "fangraphs"

    @property
    def source_detail(self) -> str:
        return f"projections/{self._projection_type}/{self._stat_type}"

    def _do_fetch(self, params: dict[str, str]) -> httpx.Response:
        response = self._client.get(_BASE_URL, params=params)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        query = {
            "type": self._projection_type,
            "stats": self._stat_type,
            "pos": "all",
            "team": "0",
            "players": "0",
        }
        logger.debug(
            "GET %s type=%s stats=%s",
            _BASE_URL,
            self._projection_type,
            self._stat_type,
        )
        response = self._fetch_with_retry(query)
        rows = _parse_response(response)
        _remap_fields(rows)
        logger.info(
            "Fetched %d FanGraphs %s projection rows (%s)",
            len(rows),
            self._projection_type,
            self._stat_type,
        )
        return rows
