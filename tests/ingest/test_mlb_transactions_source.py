import json
from typing import Any

import httpx

from fantasy_baseball_manager.ingest.mlb_transactions_source import MLBTransactionsSource


def _fake_response(transactions: list[dict[str, Any]]) -> httpx.Response:
    body = json.dumps({"transactions": transactions}).encode()
    return httpx.Response(200, content=body, headers={"content-type": "application/json"})


def _make_transaction(
    *,
    tx_id: int = 1,
    person_id: int = 545361,
    date: str = "2024-05-15T00:00:00",
    effective_date: str = "2024-05-15T00:00:00",
    description: str = "Placed on 15-day IL",
) -> dict[str, Any]:
    return {
        "id": tx_id,
        "date": date,
        "effectiveDate": effective_date,
        "description": description,
        "person": {"id": person_id},
    }


class FakeTransport(httpx.BaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._response


class TestMLBTransactionsSource:
    def test_properties(self) -> None:
        source = MLBTransactionsSource()
        assert source.source_type == "mlb_api"
        assert source.source_detail == "transactions"

    def test_valid_json_returns_correct_dataframe(self) -> None:
        txns = [
            _make_transaction(tx_id=1, person_id=545361, description="Placed on IL"),
            _make_transaction(tx_id=2, person_id=660271, description="Activated from IL"),
        ]
        response = _fake_response(txns)
        client = httpx.Client(transport=FakeTransport(response))
        source = MLBTransactionsSource(client=client)

        df = source.fetch(season=2024)

        assert len(df) == 2
        assert list(df.columns) == ["transaction_id", "mlbam_id", "date", "effective_date", "description"]
        assert df.iloc[0]["transaction_id"] == 1
        assert df.iloc[0]["mlbam_id"] == 545361
        assert df.iloc[1]["transaction_id"] == 2

    def test_empty_transactions_returns_empty_dataframe(self) -> None:
        response = _fake_response([])
        client = httpx.Client(transport=FakeTransport(response))
        source = MLBTransactionsSource(client=client)

        df = source.fetch(season=2024)

        assert len(df) == 0
        assert list(df.columns) == ["transaction_id", "mlbam_id", "date", "effective_date", "description"]

    def test_transaction_without_person_is_skipped(self) -> None:
        txns = [
            {"id": 1, "date": "2024-05-15T00:00:00", "effectiveDate": "2024-05-15T00:00:00", "description": "Trade"},
            _make_transaction(tx_id=2, person_id=660271),
        ]
        response = _fake_response(txns)
        client = httpx.Client(transport=FakeTransport(response))
        source = MLBTransactionsSource(client=client)

        df = source.fetch(season=2024)

        assert len(df) == 1
        assert df.iloc[0]["transaction_id"] == 2
