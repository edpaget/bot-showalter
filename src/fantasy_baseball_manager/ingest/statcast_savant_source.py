import csv
import datetime
import io
import logging
from typing import Any

import httpx
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from fantasy_baseball_manager.ingest._csv_helpers import nullify_empty_strings

logger = logging.getLogger(__name__)

_URL = "https://baseballsavant.mlb.com/statcast_search/csv"


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning("Retrying statcast download (attempt %d): %s", retry_state.attempt_number, retry_state.outcome)


class StatcastSavantSource:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    @property
    def source_type(self) -> str:
        return "baseball_savant"

    @property
    def source_detail(self) -> str:
        return "statcast_pitch"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _fetch_day_with_retry(self, date_str: str) -> httpx.Response:
        params = {
            "all": "true",
            "type": "details",
            "game_date_gt": date_str,
            "game_date_lt": date_str,
        }
        response = self._client.get(_URL, params=params)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        start = datetime.date.fromisoformat(params["start_dt"])
        end = datetime.date.fromisoformat(params["end_dt"])

        rows: list[dict[str, Any]] = []
        current = start
        while current <= end:
            date_str = current.isoformat()
            logger.debug("Fetching statcast data for %s", date_str)
            response = self._fetch_day_with_retry(date_str)

            reader = csv.DictReader(io.StringIO(response.text))
            day_rows = [nullify_empty_strings(row) for row in reader]
            rows.extend(day_rows)

            logger.info("Fetched %d rows for %s", len(day_rows), date_str)
            current += datetime.timedelta(days=1)

        logger.info("Total statcast rows fetched: %d", len(rows))
        return rows
