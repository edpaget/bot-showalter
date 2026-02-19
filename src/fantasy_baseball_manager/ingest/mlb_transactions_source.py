import logging
from collections.abc import Callable
from typing import Any

import httpx

from fantasy_baseball_manager.ingest._retry import default_http_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://statsapi.mlb.com/api/v1/transactions"

_DEFAULT_RETRY = default_http_retry("MLB API request")


class MLBTransactionsSource:
    def __init__(
        self,
        client: httpx.Client | None = None,
        retry: Callable[[Callable[..., Any]], Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))
        self._fetch_with_retry = retry(self._do_fetch)

    @property
    def source_type(self) -> str:
        return "mlb_api"

    @property
    def source_detail(self) -> str:
        return "transactions"

    def _do_fetch(self, params: dict[str, Any]) -> httpx.Response:
        response = self._client.get(_BASE_URL, params=params)
        response.raise_for_status()
        return response

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        season: int = params["season"]
        logger.debug("GET %s season=%d", _BASE_URL, season)
        response = self._fetch_with_retry(
            {
                "startDate": f"{season}-01-01",
                "endDate": f"{season}-12-31",
                "sportId": 1,
                "transactionTypes": "SC",
            }
        )
        logger.debug("MLB API responded %d", response.status_code)
        data = response.json()

        rows: list[dict[str, Any]] = []
        for txn in data.get("transactions", []):
            person = txn.get("person")
            if person is None:
                continue
            rows.append(
                {
                    "transaction_id": txn["id"],
                    "mlbam_id": person["id"],
                    "date": txn.get("date"),
                    "effective_date": txn.get("effectiveDate"),
                    "description": txn.get("description"),
                }
            )

        if not rows:
            return []
        logger.info("Fetched %d transaction rows for season %d", len(rows), season)
        return rows
