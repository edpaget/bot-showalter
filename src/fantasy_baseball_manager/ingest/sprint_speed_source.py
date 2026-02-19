import csv
import io
import logging
from collections.abc import Callable
from typing import Any

import httpx

from fantasy_baseball_manager.ingest._csv_helpers import nullify_empty_strings, strip_bom
from fantasy_baseball_manager.ingest._retry import default_http_retry

logger = logging.getLogger(__name__)

_URL = "https://baseballsavant.mlb.com/leaderboard/sprint_speed"

_DEFAULT_RETRY = default_http_retry("sprint speed download")


class SprintSpeedSource:
    def __init__(
        self,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
        self._fetch_with_retry = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "baseball_savant"

    @property
    def source_detail(self) -> str:
        return "sprint_speed"

    def _do_fetch(self, params: dict[str, Any]) -> httpx.Response:
        response = self._client.get(_URL, params=params)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        year = params["year"]
        min_opp = params.get("min_opp", 10)
        query = {"year": year, "position": "", "team": "", "min": min_opp, "csv": "true"}
        logger.debug("GET %s (year=%s, min_opp=%s)", _URL, year, min_opp)
        response = self._fetch_with_retry(query)
        logger.debug("Sprint speed responded %d", response.status_code)

        reader = csv.DictReader(io.StringIO(strip_bom(response.text)))
        rows: list[dict[str, Any]] = [nullify_empty_strings(row) for row in reader]

        logger.info("Parsed %d sprint speed rows for %s", len(rows), year)
        return rows
