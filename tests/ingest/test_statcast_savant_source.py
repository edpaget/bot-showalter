import httpx
import pytest

from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.ingest.statcast_savant_source import StatcastSavantSource


class FakeTransport(httpx.BaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._call_count += 1
        return self._response

    @property
    def call_count(self) -> int:
        return self._call_count


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


_CSV_HEADER = "game_pk,game_date,batter,pitcher,at_bat_number,pitch_number,pitch_type,release_speed"

_CSV_ROW = "718001,2024-06-15,545361,477132,1,1,FF,95.2"

_CSV_WITH_EMPTY = (
    "game_pk,game_date,batter,pitcher,at_bat_number,pitch_number,pitch_type,release_speed\n"
    "718001,2024-06-15,545361,477132,1,1,,\n"
)


def _csv_response(csv_text: str) -> httpx.Response:
    return httpx.Response(200, content=csv_text.encode())


class TestStatcastSavantSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = StatcastSavantSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = StatcastSavantSource()
        assert source.source_type == "baseball_savant"
        assert source.source_detail == "statcast_pitch"

    def test_single_day_returns_rows(self) -> None:
        csv_text = f"{_CSV_HEADER}\n{_CSV_ROW}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = StatcastSavantSource(client=client)

        result = source.fetch(start_dt="2024-06-15", end_dt="2024-06-15")

        assert len(result) == 1
        assert result[0]["game_pk"] == "718001"
        assert result[0]["game_date"] == "2024-06-15"
        assert result[0]["batter"] == "545361"
        assert result[0]["pitcher"] == "477132"
        assert result[0]["pitch_type"] == "FF"
        assert result[0]["release_speed"] == "95.2"

    def test_multi_day_concatenates_results(self) -> None:
        csv_text = f"{_CSV_HEADER}\n{_CSV_ROW}\n"
        transport = FakeTransport(_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = StatcastSavantSource(client=client)

        result = source.fetch(start_dt="2024-06-15", end_dt="2024-06-16")

        assert len(result) == 2
        assert transport.call_count == 2

    def test_empty_csv_day_returns_empty_list(self) -> None:
        csv_text = f"{_CSV_HEADER}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = StatcastSavantSource(client=client)

        result = source.fetch(start_dt="2024-06-15", end_dt="2024-06-15")

        assert result == []

    def test_empty_string_fields_become_none(self) -> None:
        client = httpx.Client(transport=FakeTransport(_csv_response(_CSV_WITH_EMPTY)))
        source = StatcastSavantSource(client=client)

        result = source.fetch(start_dt="2024-06-15", end_dt="2024-06-15")

        assert len(result) == 1
        assert result[0]["game_pk"] == "718001"
        assert result[0]["pitch_type"] is None
        assert result[0]["release_speed"] is None

    def test_retry_on_503_then_success(self) -> None:
        csv_text = f"{_CSV_HEADER}\n{_CSV_ROW}\n"
        transport = FailNTransport(fail_count=2, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = StatcastSavantSource(client=client)

        result = source.fetch(start_dt="2024-06-15", end_dt="2024-06-15")

        assert len(result) == 1
        assert transport.call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        csv_text = f"{_CSV_HEADER}\n{_CSV_ROW}\n"
        transport = FailNTransport(fail_count=5, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = StatcastSavantSource(client=client)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch(start_dt="2024-06-15", end_dt="2024-06-15")

        assert transport.call_count == 3

    def test_bom_is_stripped(self) -> None:
        csv_text = f"\ufeff{_CSV_HEADER}\n{_CSV_ROW}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = StatcastSavantSource(client=client)

        result = source.fetch(start_dt="2024-06-15", end_dt="2024-06-15")

        assert len(result) == 1
        assert "game_pk" in result[0]
        assert result[0]["game_pk"] == "718001"
