import csv
import io
import logging
from typing import Any

import httpx
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from fantasy_baseball_manager.ingest._csv_helpers import nullify_empty_strings, strip_bom

logger = logging.getLogger(__name__)

_URL = "https://baseballsavant.mlb.com/leaderboard/sprint_speed"


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning("Retrying sprint speed download (attempt %d): %s", retry_state.attempt_number, retry_state.outcome)


class SprintSpeedSource:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))

    @property
    def source_type(self) -> str:
        return "baseball_savant"

    @property
    def source_detail(self) -> str:
        return "sprint_speed"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _fetch_with_retry(self, params: dict[str, Any]) -> httpx.Response:
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
