import logging
from collections.abc import Callable
from typing import Any

import httpx

from fantasy_baseball_manager.ingest._retry import default_http_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.fangraphs.com/api/leaders/major-league/data"

_DEFAULT_RETRY = default_http_retry("FanGraphs API call")


def _parse_response(response: httpx.Response) -> list[dict[str, Any]]:
    """Parse JSON response, handling both bare array and {"data": [...]} formats."""
    data = response.json()
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data


def _remap_playerid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remap ``playerid`` → ``IDfg`` so downstream mappers work unchanged."""
    for row in rows:
        if "playerid" in row:
            row["IDfg"] = row["playerid"]
    return rows


class FgBattingSource:
    def __init__(
        self,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._fetch_with_retry = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "fangraphs"

    @property
    def source_detail(self) -> str:
        return "batting"

    def _do_fetch(self, params: dict[str, str]) -> httpx.Response:
        response = self._client.get(_BASE_URL, params=params)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        season = params["season"]
        qual = params.get("qual", 0)
        query = {
            "stats": "bat",
            "season": str(season),
            "season1": str(season),
            "ind": "1",
            "qual": str(qual),
            "type": "8",
            "pageitems": "2000000",
            "lg": "all",
            "pos": "all",
            "month": "0",
            "team": "0",
            "age": "",
            "hand": "",
            "pagenum": "1",
        }
        logger.debug("GET %s stats=bat season=%s", _BASE_URL, season)
        response = self._fetch_with_retry(query)
        rows = _parse_response(response)
        _remap_playerid(rows)
        logger.info("Fetched %d FanGraphs batting rows for %s", len(rows), season)
        return rows


class FgPitchingSource:
    def __init__(
        self,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._fetch_with_retry = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "fangraphs"

    @property
    def source_detail(self) -> str:
        return "pitching"

    def _do_fetch(self, params: dict[str, str]) -> httpx.Response:
        response = self._client.get(_BASE_URL, params=params)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        season = params["season"]
        qual = params.get("qual", 0)
        query = {
            "stats": "pit",
            "season": str(season),
            "season1": str(season),
            "ind": "1",
            "qual": str(qual),
            "type": "8",
            "pageitems": "2000000",
            "lg": "all",
            "pos": "all",
            "month": "0",
            "team": "0",
            "age": "",
            "hand": "",
            "pagenum": "1",
        }
        logger.debug("GET %s stats=pit season=%s", _BASE_URL, season)
        response = self._fetch_with_retry(query)
        rows = _parse_response(response)
        _remap_playerid(rows)
        logger.info("Fetched %d FanGraphs pitching rows for %s", len(rows), season)
        return rows
