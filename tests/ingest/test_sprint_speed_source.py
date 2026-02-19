import httpx
import pytest
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.ingest.sprint_speed_source import SprintSpeedSource

_NO_WAIT_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_none(),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    reraise=True,
)


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


_CSV_HEADER = "player_id,sprint_speed,hp_to_1b,bolts,competitive_runs"

_CSV_ROWS = "545361,28.5,4.18,3,50\n660271,29.1,4.09,5,65\n"

_CSV_WITH_EMPTY = "player_id,sprint_speed,hp_to_1b,bolts,competitive_runs\n545361,,4.18,,50\n"


def _csv_response(csv_text: str) -> httpx.Response:
    return httpx.Response(200, content=csv_text.encode())


class TestSprintSpeedSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = SprintSpeedSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = SprintSpeedSource()
        assert source.source_type == "baseball_savant"
        assert source.source_detail == "sprint_speed"

    def test_valid_csv_returns_list_of_dicts(self) -> None:
        csv_text = f"{_CSV_HEADER}\n{_CSV_ROWS}"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = SprintSpeedSource(client=client)

        result = source.fetch(year=2024)

        assert len(result) == 2
        assert result[0]["player_id"] == "545361"
        assert result[0]["sprint_speed"] == "28.5"
        assert result[0]["hp_to_1b"] == "4.18"
        assert result[0]["bolts"] == "3"
        assert result[0]["competitive_runs"] == "50"
        assert result[1]["player_id"] == "660271"

    def test_empty_string_fields_become_none(self) -> None:
        client = httpx.Client(transport=FakeTransport(_csv_response(_CSV_WITH_EMPTY)))
        source = SprintSpeedSource(client=client)

        result = source.fetch(year=2024)

        assert len(result) == 1
        assert result[0]["player_id"] == "545361"
        assert result[0]["sprint_speed"] is None
        assert result[0]["hp_to_1b"] == "4.18"
        assert result[0]["bolts"] is None
        assert result[0]["competitive_runs"] == "50"

    def test_empty_csv_returns_empty_list(self) -> None:
        csv_text = f"{_CSV_HEADER}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = SprintSpeedSource(client=client)

        result = source.fetch(year=2024)

        assert result == []

    def test_retry_on_503_then_success(self) -> None:
        csv_text = f"{_CSV_HEADER}\n{_CSV_ROWS}"
        transport = FailNTransport(fail_count=2, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = SprintSpeedSource(client=client, retry=_NO_WAIT_RETRY)

        result = source.fetch(year=2024)

        assert len(result) == 2
        assert transport.call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        csv_text = f"{_CSV_HEADER}\n{_CSV_ROWS}"
        transport = FailNTransport(fail_count=5, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = SprintSpeedSource(client=client, retry=_NO_WAIT_RETRY)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch(year=2024)

        assert transport.call_count == 3

    def test_bom_is_stripped(self) -> None:
        csv_text = f"\ufeff{_CSV_HEADER}\n{_CSV_ROWS}"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = SprintSpeedSource(client=client)

        result = source.fetch(year=2024)

        assert len(result) == 2
        assert "player_id" in result[0]
        assert result[0]["player_id"] == "545361"
