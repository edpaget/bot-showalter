import json
from typing import Any

import httpx
import pytest
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

from fantasy_baseball_manager.ingest.mlb_transactions_source import MLBTransactionsSource

_NO_WAIT_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_none(),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    reraise=True,
)


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


class FailNTransport(httpx.BaseTransport):
    """Returns error responses for the first N requests, then succeeds."""

    def __init__(self, fail_count: int, success_response: httpx.Response) -> None:
        self._fail_count = fail_count
        self._success_response = success_response
        self._call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            return httpx.Response(503, content=b"Service Unavailable")
        return self._success_response

    @property
    def call_count(self) -> int:
        return self._call_count


class TestMLBTransactionsSource:
    def test_properties(self) -> None:
        source = MLBTransactionsSource()
        assert source.source_type == "mlb_api"
        assert source.source_detail == "transactions"

    def test_valid_json_returns_list_of_dicts(self) -> None:
        txns = [
            _make_transaction(tx_id=1, person_id=545361, description="Placed on IL"),
            _make_transaction(tx_id=2, person_id=660271, description="Activated from IL"),
        ]
        response = _fake_response(txns)
        client = httpx.Client(transport=FakeTransport(response))
        source = MLBTransactionsSource(client=client)

        result = source.fetch(season=2024)

        assert len(result) == 2
        assert list(result[0].keys()) == ["transaction_id", "mlbam_id", "date", "effective_date", "description"]
        assert result[0]["transaction_id"] == 1
        assert result[0]["mlbam_id"] == 545361
        assert result[1]["transaction_id"] == 2

    def test_empty_transactions_returns_empty_list(self) -> None:
        response = _fake_response([])
        client = httpx.Client(transport=FakeTransport(response))
        source = MLBTransactionsSource(client=client)

        result = source.fetch(season=2024)

        assert result == []

    def test_transaction_without_person_is_skipped(self) -> None:
        txns = [
            {"id": 1, "date": "2024-05-15T00:00:00", "effectiveDate": "2024-05-15T00:00:00", "description": "Trade"},
            _make_transaction(tx_id=2, person_id=660271),
        ]
        response = _fake_response(txns)
        client = httpx.Client(transport=FakeTransport(response))
        source = MLBTransactionsSource(client=client)

        result = source.fetch(season=2024)

        assert len(result) == 1
        assert result[0]["transaction_id"] == 2

    def test_retry_on_503_then_success(self) -> None:
        txns = [_make_transaction()]
        success_response = _fake_response(txns)
        transport = FailNTransport(fail_count=2, success_response=success_response)
        client = httpx.Client(transport=transport)
        source = MLBTransactionsSource(client=client, retry=_NO_WAIT_RETRY)

        result = source.fetch(season=2024)

        assert len(result) == 1
        assert transport.call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        txns = [_make_transaction()]
        success_response = _fake_response(txns)
        transport = FailNTransport(fail_count=5, success_response=success_response)
        client = httpx.Client(transport=transport)
        source = MLBTransactionsSource(client=client, retry=_NO_WAIT_RETRY)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch(season=2024)

        assert transport.call_count == 3
