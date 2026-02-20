import httpx
import pytest
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

from fantasy_baseball_manager.ingest.lahman_source import (
    LahmanAppearancesSource,
    LahmanCsvSource,
    LahmanPeopleSource,
    LahmanTeamsSource,
)
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


def _csv_response(csv_text: str) -> httpx.Response:
    return httpx.Response(200, content=csv_text.encode())


# --- People CSV test data ---

_PEOPLE_HEADER = "ID,playerID,retroID,birthYear,birthMonth,birthDay,bats,throws"
_PEOPLE_ROWS = "1,troutmi01,troum001,1991,8,7,R,R\n2,ohtansh01,ohtas001,1994,7,5,L,R\n"
_PEOPLE_WITH_EMPTY = "ID,playerID,retroID,birthYear,birthMonth,birthDay,bats,throws\n1,troutmi01,troum001,,,,,\n"

# --- Appearances CSV test data ---

_APP_HEADER = "playerID,yearID,teamID,G_p,G_c,G_1b,G_2b,G_3b,G_ss,G_lf,G_cf,G_rf,G_dh"
_APP_ROW_2023 = "troutmi01,2023,LAA,0,0,0,0,0,0,0,82,0,25"
_APP_ROW_2022 = "ohtansh01,2022,LAA,10,0,0,0,0,0,0,0,0,100"
_APP_ROW_ZEROS = "nobody01,2023,NYY,0,0,0,0,0,0,0,0,0,0"
_APP_ROW_EMPTY = "empty01,2023,BOS,,,,,,,,,,"

# --- Teams CSV test data ---

_TEAMS_HEADER = "teamID,yearID,lgID,divID,name"
_TEAMS_ROW_2023 = "LAA,2023,AL,W,Los Angeles Angels"
_TEAMS_ROW_2022 = "NYY,2022,AL,E,New York Yankees"
_TEAMS_ROW_EMPTY = "BOS,2023,,,Boston Red Sox"


class TestLahmanPeopleSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = LahmanPeopleSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = LahmanPeopleSource()
        assert source.source_type == "lahman"
        assert source.source_detail == "people"

    def test_valid_csv_returns_list_of_dicts(self) -> None:
        csv_text = f"{_PEOPLE_HEADER}\n{_PEOPLE_ROWS}"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanPeopleSource(client=client)

        result = source.fetch()

        assert len(result) == 2
        assert result[0]["playerID"] == "troutmi01"
        assert result[0]["retroID"] == "troum001"
        assert result[0]["birthYear"] == "1991"
        assert result[0]["birthMonth"] == "8"
        assert result[0]["birthDay"] == "7"
        assert result[0]["bats"] == "R"
        assert result[0]["throws"] == "R"
        # ID column is present (not filtered)
        assert result[0]["ID"] == "1"

    def test_empty_string_fields_become_none(self) -> None:
        client = httpx.Client(transport=FakeTransport(_csv_response(_PEOPLE_WITH_EMPTY)))
        source = LahmanPeopleSource(client=client)

        result = source.fetch()

        assert len(result) == 1
        assert result[0]["playerID"] == "troutmi01"
        assert result[0]["birthYear"] is None
        assert result[0]["birthMonth"] is None
        assert result[0]["birthDay"] is None
        assert result[0]["bats"] is None
        assert result[0]["throws"] is None

    def test_empty_csv_returns_empty_list(self) -> None:
        csv_text = f"{_PEOPLE_HEADER}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanPeopleSource(client=client)

        result = source.fetch()

        assert result == []

    def test_bom_is_stripped(self) -> None:
        csv_text = f"\ufeff{_PEOPLE_HEADER}\n{_PEOPLE_ROWS}"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanPeopleSource(client=client)

        result = source.fetch()

        assert len(result) == 2
        assert "ID" in result[0]
        assert result[0]["ID"] == "1"

    def test_retry_on_503_then_success(self) -> None:
        csv_text = f"{_PEOPLE_HEADER}\n{_PEOPLE_ROWS}"
        transport = FailNTransport(fail_count=2, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = LahmanPeopleSource(client=client, retry=_NO_WAIT_RETRY)

        result = source.fetch()

        assert len(result) == 2
        assert transport.call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        csv_text = f"{_PEOPLE_HEADER}\n{_PEOPLE_ROWS}"
        transport = FailNTransport(fail_count=5, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = LahmanPeopleSource(client=client, retry=_NO_WAIT_RETRY)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch()

        assert transport.call_count == 3


class TestLahmanAppearancesSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = LahmanAppearancesSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = LahmanAppearancesSource()
        assert source.source_type == "lahman"
        assert source.source_detail == "appearances"

    def test_explodes_wide_to_long(self) -> None:
        csv_text = f"{_APP_HEADER}\n{_APP_ROW_2023}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanAppearancesSource(client=client)

        result = source.fetch(season=2023)

        # G_cf=82, G_dh=25 => 2 rows (others are 0)
        assert len(result) == 2
        positions = {r["position"] for r in result}
        assert positions == {"CF", "DH"}
        for r in result:
            assert r["playerID"] == "troutmi01"
            assert r["yearID"] == 2023
            assert r["teamID"] == "LAA"
            assert "position" in r
            assert "games" in r
        cf_row = next(r for r in result if r["position"] == "CF")
        assert cf_row["games"] == 82
        dh_row = next(r for r in result if r["position"] == "DH")
        assert dh_row["games"] == 25

    def test_zero_games_excluded(self) -> None:
        csv_text = f"{_APP_HEADER}\n{_APP_ROW_ZEROS}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanAppearancesSource(client=client)

        result = source.fetch(season=2023)

        assert result == []

    def test_empty_games_excluded(self) -> None:
        csv_text = f"{_APP_HEADER}\n{_APP_ROW_EMPTY}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanAppearancesSource(client=client)

        result = source.fetch(season=2023)

        assert result == []

    def test_season_filter(self) -> None:
        csv_text = f"{_APP_HEADER}\n{_APP_ROW_2023}\n{_APP_ROW_2022}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanAppearancesSource(client=client)

        result = source.fetch(season=2023)

        for r in result:
            assert r["yearID"] == 2023

    def test_no_season_returns_all(self) -> None:
        csv_text = f"{_APP_HEADER}\n{_APP_ROW_2023}\n{_APP_ROW_2022}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanAppearancesSource(client=client)

        result = source.fetch()

        years = {r["yearID"] for r in result}
        assert 2022 in years
        assert 2023 in years

    def test_empty_csv_returns_empty_list(self) -> None:
        csv_text = f"{_APP_HEADER}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanAppearancesSource(client=client)

        result = source.fetch(season=2023)

        assert result == []

    def test_bom_is_stripped(self) -> None:
        csv_text = f"\ufeff{_APP_HEADER}\n{_APP_ROW_2023}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanAppearancesSource(client=client)

        result = source.fetch(season=2023)

        assert len(result) == 2

    def test_retry_on_503_then_success(self) -> None:
        csv_text = f"{_APP_HEADER}\n{_APP_ROW_2023}\n"
        transport = FailNTransport(fail_count=2, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = LahmanAppearancesSource(client=client, retry=_NO_WAIT_RETRY)

        result = source.fetch(season=2023)

        assert len(result) == 2
        assert transport.call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        csv_text = f"{_APP_HEADER}\n{_APP_ROW_2023}\n"
        transport = FailNTransport(fail_count=5, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = LahmanAppearancesSource(client=client, retry=_NO_WAIT_RETRY)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch(season=2023)

        assert transport.call_count == 3


class TestLahmanTeamsSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = LahmanTeamsSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = LahmanTeamsSource()
        assert source.source_type == "lahman"
        assert source.source_detail == "teams"

    def test_valid_csv_returns_list_of_dicts(self) -> None:
        csv_text = f"{_TEAMS_HEADER}\n{_TEAMS_ROW_2023}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanTeamsSource(client=client)

        result = source.fetch(season=2023)

        assert len(result) == 1
        assert result[0]["teamID"] == "LAA"
        assert result[0]["yearID"] == "2023"
        assert result[0]["lgID"] == "AL"
        assert result[0]["divID"] == "W"
        assert result[0]["name"] == "Los Angeles Angels"

    def test_season_filter(self) -> None:
        csv_text = f"{_TEAMS_HEADER}\n{_TEAMS_ROW_2023}\n{_TEAMS_ROW_2022}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanTeamsSource(client=client)

        result = source.fetch(season=2023)

        assert len(result) == 1
        assert result[0]["teamID"] == "LAA"

    def test_no_season_returns_all(self) -> None:
        csv_text = f"{_TEAMS_HEADER}\n{_TEAMS_ROW_2023}\n{_TEAMS_ROW_2022}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanTeamsSource(client=client)

        result = source.fetch()

        assert len(result) == 2

    def test_empty_string_fields_become_none(self) -> None:
        csv_text = f"{_TEAMS_HEADER}\n{_TEAMS_ROW_EMPTY}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanTeamsSource(client=client)

        result = source.fetch(season=2023)

        assert len(result) == 1
        assert result[0]["teamID"] == "BOS"
        assert result[0]["lgID"] is None
        assert result[0]["divID"] is None

    def test_empty_csv_returns_empty_list(self) -> None:
        csv_text = f"{_TEAMS_HEADER}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanTeamsSource(client=client)

        result = source.fetch(season=2023)

        assert result == []

    def test_bom_is_stripped(self) -> None:
        csv_text = f"\ufeff{_TEAMS_HEADER}\n{_TEAMS_ROW_2023}\n"
        client = httpx.Client(transport=FakeTransport(_csv_response(csv_text)))
        source = LahmanTeamsSource(client=client)

        result = source.fetch(season=2023)

        assert len(result) == 1
        assert "teamID" in result[0]

    def test_retry_on_503_then_success(self) -> None:
        csv_text = f"{_TEAMS_HEADER}\n{_TEAMS_ROW_2023}\n"
        transport = FailNTransport(fail_count=2, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = LahmanTeamsSource(client=client, retry=_NO_WAIT_RETRY)

        result = source.fetch(season=2023)

        assert len(result) == 1
        assert transport.call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        csv_text = f"{_TEAMS_HEADER}\n{_TEAMS_ROW_2023}\n"
        transport = FailNTransport(fail_count=5, success_response=_csv_response(csv_text))
        client = httpx.Client(transport=transport)
        source = LahmanTeamsSource(client=client, retry=_NO_WAIT_RETRY)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch(season=2023)

        assert transport.call_count == 3


class TestLahmanCsvSourceBase:
    def test_people_is_subclass(self) -> None:
        assert issubclass(LahmanPeopleSource, LahmanCsvSource)

    def test_appearances_is_subclass(self) -> None:
        assert issubclass(LahmanAppearancesSource, LahmanCsvSource)

    def test_teams_is_subclass(self) -> None:
        assert issubclass(LahmanTeamsSource, LahmanCsvSource)
