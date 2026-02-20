import httpx
import pytest
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

from fantasy_baseball_manager.ingest.fangraphs_source import FgStatsSource
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

_STAT_TYPE_CASES = [
    ("bat", "batting", _BATTING_ROWS, "PA", 600, 10155),
    ("pit", "pitching", _PITCHING_ROWS, "W", 15, 19755),
]


class TestFgStatsSource:
    @pytest.mark.parametrize(("stat_type", "detail", "rows", "sample_key", "sample_val", "idfg"), _STAT_TYPE_CASES)
    def test_satisfies_datasource_protocol(
        self, stat_type: str, detail: str, rows: list[dict], sample_key: str, sample_val: object, idfg: int
    ) -> None:
        source = FgStatsSource(stat_type=stat_type)
        assert isinstance(source, DataSource)

    @pytest.mark.parametrize(("stat_type", "detail", "rows", "sample_key", "sample_val", "idfg"), _STAT_TYPE_CASES)
    def test_source_type_and_detail(
        self, stat_type: str, detail: str, rows: list[dict], sample_key: str, sample_val: object, idfg: int
    ) -> None:
        source = FgStatsSource(stat_type=stat_type)
        assert source.source_type == "fangraphs"
        assert source.source_detail == detail

    @pytest.mark.parametrize(("stat_type", "detail", "rows", "sample_key", "sample_val", "idfg"), _STAT_TYPE_CASES)
    def test_valid_json_returns_list_of_dicts(
        self, stat_type: str, detail: str, rows: list[dict], sample_key: str, sample_val: object, idfg: int
    ) -> None:
        transport = FakeTransport(_json_response(rows))
        client = httpx.Client(transport=transport)
        source = FgStatsSource(stat_type=stat_type, client=client)

        result = source.fetch(season=2024)

        assert len(result) == 1
        assert result[0][sample_key] == sample_val

    @pytest.mark.parametrize(("stat_type", "detail", "rows", "sample_key", "sample_val", "idfg"), _STAT_TYPE_CASES)
    def test_playerid_remapped_to_idfg(
        self, stat_type: str, detail: str, rows: list[dict], sample_key: str, sample_val: object, idfg: int
    ) -> None:
        transport = FakeTransport(_json_response(rows))
        client = httpx.Client(transport=transport)
        source = FgStatsSource(stat_type=stat_type, client=client)

        result = source.fetch(season=2024)

        assert result[0]["IDfg"] == idfg

    @pytest.mark.parametrize(("stat_type", "detail", "rows", "sample_key", "sample_val", "idfg"), _STAT_TYPE_CASES)
    def test_empty_response_returns_empty_list(
        self, stat_type: str, detail: str, rows: list[dict], sample_key: str, sample_val: object, idfg: int
    ) -> None:
        transport = FakeTransport(_json_response([]))
        client = httpx.Client(transport=transport)
        source = FgStatsSource(stat_type=stat_type, client=client)

        result = source.fetch(season=2024)

        assert result == []

    @pytest.mark.parametrize(("stat_type", "detail", "rows", "sample_key", "sample_val", "idfg"), _STAT_TYPE_CASES)
    def test_query_params_include_season(
        self, stat_type: str, detail: str, rows: list[dict], sample_key: str, sample_val: object, idfg: int
    ) -> None:
        transport = FakeTransport(_json_response([]))
        client = httpx.Client(transport=transport)
        source = FgStatsSource(stat_type=stat_type, client=client)

        source.fetch(season=2024)

        assert transport.last_request is not None
        params = dict(transport.last_request.url.params)
        assert params["stats"] == stat_type
        assert params["season"] == "2024"
        assert params["season1"] == "2024"
        assert params["qual"] == "0"
        assert params["ind"] == "1"
        assert params["type"] == "8"
        assert params["pageitems"] == "2000000"

    @pytest.mark.parametrize(("stat_type", "detail", "rows", "sample_key", "sample_val", "idfg"), _STAT_TYPE_CASES)
    def test_retry_on_503_then_success(
        self, stat_type: str, detail: str, rows: list[dict], sample_key: str, sample_val: object, idfg: int
    ) -> None:
        transport = FailNTransport(fail_count=2, success_response=_json_response(rows))
        client = httpx.Client(transport=transport)
        source = FgStatsSource(stat_type=stat_type, client=client, retry=_NO_WAIT_RETRY)

        result = source.fetch(season=2024)

        assert len(result) == 1
        assert transport.call_count == 3

    @pytest.mark.parametrize(("stat_type", "detail", "rows", "sample_key", "sample_val", "idfg"), _STAT_TYPE_CASES)
    def test_exhausted_retries_raises(
        self, stat_type: str, detail: str, rows: list[dict], sample_key: str, sample_val: object, idfg: int
    ) -> None:
        transport = FailNTransport(fail_count=5, success_response=_json_response(rows))
        client = httpx.Client(transport=transport)
        source = FgStatsSource(stat_type=stat_type, client=client, retry=_NO_WAIT_RETRY)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch(season=2024)

        assert transport.call_count == 3

    @pytest.mark.parametrize(("stat_type", "detail", "rows", "sample_key", "sample_val", "idfg"), _STAT_TYPE_CASES)
    def test_data_wrapper_response(
        self, stat_type: str, detail: str, rows: list[dict], sample_key: str, sample_val: object, idfg: int
    ) -> None:
        """API may return {"data": [...]} instead of bare array."""
        transport = FakeTransport(_json_response({"data": rows}))
        client = httpx.Client(transport=transport)
        source = FgStatsSource(stat_type=stat_type, client=client)

        result = source.fetch(season=2024)

        assert len(result) == 1
        assert result[0]["IDfg"] == idfg

    def test_invalid_stat_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown stat_type 'invalid'"):
            FgStatsSource(stat_type="invalid")
