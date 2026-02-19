import csv
import datetime
import io
import logging
from collections.abc import Callable
from typing import Any

import httpx

from fantasy_baseball_manager.ingest._csv_helpers import nullify_empty_strings, strip_bom
from fantasy_baseball_manager.ingest._retry import default_http_retry

logger = logging.getLogger(__name__)

_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

_DEFAULT_RETRY = default_http_retry("statcast download")


class StatcastSavantSource:
    def __init__(
        self,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        self._fetch_day_with_retry = retry(self._do_fetch_day)

    @property
    def source_type(self) -> str:
        return "baseball_savant"

    @property
    def source_detail(self) -> str:
        return "statcast_pitch"

    def _do_fetch_day(self, date_str: str) -> httpx.Response:
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

            reader = csv.DictReader(io.StringIO(strip_bom(response.text)))
            day_rows = [nullify_empty_strings(row) for row in reader]
            rows.extend(day_rows)

            logger.info("Fetched %d rows for %s", len(day_rows), date_str)
            current += datetime.timedelta(days=1)

        logger.info("Total statcast rows fetched: %d", len(rows))
        return rows
