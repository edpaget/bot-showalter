import httpx
import pytest
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

from fantasy_baseball_manager.ingest.fangraphs_source import FgBattingSource, FgPitchingSource
from fantasy_baseball_manager.ingest.protocols import DataSource

_NO_WAIT_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_none(),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    reraise=True,
)


class FakeTransport(httpx.BaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
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


def _json_response(data: object) -> httpx.Response:
    return httpx.Response(200, json=data)


# --- Batting test data ---

_BATTING_ROWS = [
    {
        "playerid": 10155,
        "Season": 2024,
        "PA": 600,
        "AB": 530,
        "H": 160,
        "2B": 30,
        "3B": 5,
        "HR": 35,
        "RBI": 90,
        "R": 100,
        "SB": 15,
        "CS": 3,
        "BB": 60,
        "SO": 120,
        "AVG": 0.302,
        "wOBA": 0.410,
        "wRC+": 170.0,
        "WAR": 8.5,
    },
]

# --- Pitching test data ---

_PITCHING_ROWS = [
    {
        "playerid": 19755,
        "Season": 2024,
        "W": 15,
        "L": 5,
        "G": 30,
        "GS": 30,
        "SV": 0,
        "IP": 180.0,
        "H": 120,
        "ER": 50,
        "HR": 15,
        "BB": 40,
        "SO": 200,
        "ERA": 2.80,
        "WHIP": 0.95,
        "K/9": 10.0,
        "BB/9": 2.0,
        "FIP": 2.90,
        "xFIP": 3.00,
        "WAR": 6.0,
    },
]


class TestFgBattingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = FgBattingSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = FgBattingSource()
        assert source.source_type == "fangraphs"
        assert source.source_detail == "batting"

    def test_valid_json_returns_list_of_dicts(self) -> None:
        transport = FakeTransport(_json_response(_BATTING_ROWS))
        client = httpx.Client(transport=transport)
        source = FgBattingSource(client=client)

        result = source.fetch(season=2024)

        assert len(result) == 1
        row = result[0]
        assert row["PA"] == 600
        assert row["AB"] == 530
        assert row["HR"] == 35
        assert row["AVG"] == 0.302
        assert row["wOBA"] == 0.410
        assert row["wRC+"] == 170.0
        assert row["WAR"] == 8.5

    def test_playerid_remapped_to_idfg(self) -> None:
        transport = FakeTransport(_json_response(_BATTING_ROWS))
        client = httpx.Client(transport=transport)
        source = FgBattingSource(client=client)

        result = source.fetch(season=2024)

        assert result[0]["IDfg"] == 10155

    def test_empty_response_returns_empty_list(self) -> None:
        transport = FakeTransport(_json_response([]))
        client = httpx.Client(transport=transport)
        source = FgBattingSource(client=client)

        result = source.fetch(season=2024)

        assert result == []

    def test_query_params_include_season(self) -> None:
        transport = FakeTransport(_json_response([]))
        client = httpx.Client(transport=transport)
        source = FgBattingSource(client=client)

        source.fetch(season=2024)

        assert transport.last_request is not None
        params = dict(transport.last_request.url.params)
        assert params["stats"] == "bat"
        assert params["season"] == "2024"
        assert params["season1"] == "2024"
        assert params["qual"] == "0"
        assert params["ind"] == "1"
        assert params["type"] == "8"
        assert params["pageitems"] == "2000000"

    def test_retry_on_503_then_success(self) -> None:
        transport = FailNTransport(fail_count=2, success_response=_json_response(_BATTING_ROWS))
        client = httpx.Client(transport=transport)
        source = FgBattingSource(client=client, retry=_NO_WAIT_RETRY)

        result = source.fetch(season=2024)

        assert len(result) == 1
        assert transport.call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        transport = FailNTransport(fail_count=5, success_response=_json_response(_BATTING_ROWS))
        client = httpx.Client(transport=transport)
        source = FgBattingSource(client=client, retry=_NO_WAIT_RETRY)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch(season=2024)

        assert transport.call_count == 3

    def test_data_wrapper_response(self) -> None:
        """API may return {"data": [...]} instead of bare array."""
        transport = FakeTransport(_json_response({"data": _BATTING_ROWS}))
        client = httpx.Client(transport=transport)
        source = FgBattingSource(client=client)

        result = source.fetch(season=2024)

        assert len(result) == 1
        assert result[0]["IDfg"] == 10155


class TestFgPitchingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = FgPitchingSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = FgPitchingSource()
        assert source.source_type == "fangraphs"
        assert source.source_detail == "pitching"

    def test_valid_json_returns_list_of_dicts(self) -> None:
        transport = FakeTransport(_json_response(_PITCHING_ROWS))
        client = httpx.Client(transport=transport)
        source = FgPitchingSource(client=client)

        result = source.fetch(season=2024)

        assert len(result) == 1
        row = result[0]
        assert row["W"] == 15
        assert row["SO"] == 200
        assert row["ERA"] == 2.80
        assert row["IP"] == 180.0
        assert row["WAR"] == 6.0

    def test_playerid_remapped_to_idfg(self) -> None:
        transport = FakeTransport(_json_response(_PITCHING_ROWS))
        client = httpx.Client(transport=transport)
        source = FgPitchingSource(client=client)

        result = source.fetch(season=2024)

        assert result[0]["IDfg"] == 19755

    def test_empty_response_returns_empty_list(self) -> None:
        transport = FakeTransport(_json_response([]))
        client = httpx.Client(transport=transport)
        source = FgPitchingSource(client=client)

        result = source.fetch(season=2024)

        assert result == []

    def test_query_params_include_season(self) -> None:
        transport = FakeTransport(_json_response([]))
        client = httpx.Client(transport=transport)
        source = FgPitchingSource(client=client)

        source.fetch(season=2024)

        assert transport.last_request is not None
        params = dict(transport.last_request.url.params)
        assert params["stats"] == "pit"
        assert params["season"] == "2024"
        assert params["season1"] == "2024"

    def test_retry_on_503_then_success(self) -> None:
        transport = FailNTransport(fail_count=2, success_response=_json_response(_PITCHING_ROWS))
        client = httpx.Client(transport=transport)
        source = FgPitchingSource(client=client, retry=_NO_WAIT_RETRY)

        result = source.fetch(season=2024)

        assert len(result) == 1
        assert transport.call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        transport = FailNTransport(fail_count=5, success_response=_json_response(_PITCHING_ROWS))
        client = httpx.Client(transport=transport)
        source = FgPitchingSource(client=client, retry=_NO_WAIT_RETRY)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch(season=2024)

        assert transport.call_count == 3

    def test_data_wrapper_response(self) -> None:
        """API may return {"data": [...]} instead of bare array."""
        transport = FakeTransport(_json_response({"data": _PITCHING_ROWS}))
        client = httpx.Client(transport=transport)
        source = FgPitchingSource(client=client)

        result = source.fetch(season=2024)

        assert len(result) == 1
        assert result[0]["IDfg"] == 19755
