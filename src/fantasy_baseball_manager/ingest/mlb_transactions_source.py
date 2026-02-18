import logging
from typing import Any

import httpx
from tenacity import RetryCallState, retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)


def _log_retry(retry_state: RetryCallState) -> None:
    logger.warning("Retrying MLB API request (attempt %d): %s", retry_state.attempt_number, retry_state.outcome)


_BASE_URL = "https://statsapi.mlb.com/api/v1/transactions"


class MLBTransactionsSource:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))

    @property
    def source_type(self) -> str:
        return "mlb_api"

    @property
    def source_detail(self) -> str:
        return "transactions"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        before_sleep=_log_retry,
        reraise=True,
    )
    def _fetch_with_retry(self, params: dict[str, Any]) -> httpx.Response:
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
