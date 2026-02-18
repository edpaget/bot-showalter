import logging
from typing import Any

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://statsapi.mlb.com/api/v1/transactions"
_COLUMNS = ["transaction_id", "mlbam_id", "date", "effective_date", "description"]


class MLBTransactionsSource:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client()

    @property
    def source_type(self) -> str:
        return "mlb_api"

    @property
    def source_detail(self) -> str:
        return "transactions"

    def fetch(self, **params: Any) -> pd.DataFrame:
        season: int = params["season"]
        logger.debug("GET %s season=%d", _BASE_URL, season)
        response = self._client.get(
            _BASE_URL,
            params={
                "startDate": f"{season}-01-01",
                "endDate": f"{season}-12-31",
                "sportId": 1,
                "transactionTypes": "SC",
            },
        )
        response.raise_for_status()
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
            return pd.DataFrame(columns=_COLUMNS)
        logger.info("Fetched %d transaction rows for season %d", len(rows), season)
        return pd.DataFrame(rows)
