from typing import TYPE_CHECKING, Any

import httpx
import pytest
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.result import Ok
from fantasy_baseball_manager.ingest.column_maps import (
    make_fg_projection_batting_mapper,
    make_fg_projection_pitching_mapper,
)
from fantasy_baseball_manager.ingest.fangraphs_projection_source import (
    PROJECTION_SYSTEMS,
    FgProjectionSource,
)
from fantasy_baseball_manager.ingest.loader import Loader
from fantasy_baseball_manager.ingest.protocols import DataSource
from fantasy_baseball_manager.repos.load_log_repo import SqliteLoadLogRepo
from fantasy_baseball_manager.repos.player_repo import SqlitePlayerRepo
from fantasy_baseball_manager.repos.projection_repo import SqliteProjectionRepo
from tests.helpers import seed_player

if TYPE_CHECKING:
    import sqlite3

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


_BATTING_ROWS = [
    {
        "playerid": "10155",
        "xMLBAMID": 545361,
        "PlayerName": "Mike Trout",
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

_PITCHING_ROWS = [
    {
        "playerid": "19755",
        "xMLBAMID": 669373,
        "PlayerName": "Corbin Burnes",
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
        "WAR": 6.0,
    },
]

_STAT_TYPE_CASES = [
    ("bat", "projections/fangraphsdc/bat", _BATTING_ROWS, "PA", 600, "10155", 545361),
    ("pit", "projections/fangraphsdc/pit", _PITCHING_ROWS, "W", 15, "19755", 669373),
]


class TestFgProjectionSource:
    @pytest.mark.parametrize(
        ("stat_type", "detail", "rows", "sample_key", "sample_val", "player_id", "mlbam_id"),
        _STAT_TYPE_CASES,
    )
    def test_satisfies_datasource_protocol(
        self,
        stat_type: str,
        detail: str,
        rows: list[dict],
        sample_key: str,
        sample_val: object,
        player_id: str,
        mlbam_id: int,
    ) -> None:
        source = FgProjectionSource(projection_type="fangraphsdc", stat_type=stat_type)
        assert isinstance(source, DataSource)

    @pytest.mark.parametrize(
        ("stat_type", "detail", "rows", "sample_key", "sample_val", "player_id", "mlbam_id"),
        _STAT_TYPE_CASES,
    )
    def test_source_type_and_detail(
        self,
        stat_type: str,
        detail: str,
        rows: list[dict],
        sample_key: str,
        sample_val: object,
        player_id: str,
        mlbam_id: int,
    ) -> None:
        source = FgProjectionSource(projection_type="fangraphsdc", stat_type=stat_type)
        assert source.source_type == "fangraphs"
        assert source.source_detail == detail

    @pytest.mark.parametrize(
        ("stat_type", "detail", "rows", "sample_key", "sample_val", "player_id", "mlbam_id"),
        _STAT_TYPE_CASES,
    )
    def test_fetch_returns_rows_with_remapped_fields(
        self,
        stat_type: str,
        detail: str,
        rows: list[dict],
        sample_key: str,
        sample_val: object,
        player_id: str,
        mlbam_id: int,
    ) -> None:
        transport = FakeTransport(_json_response(rows))
        client = httpx.Client(transport=transport)
        source = FgProjectionSource(projection_type="fangraphsdc", stat_type=stat_type, client=client)

        result = source.fetch()

        assert len(result) == 1
        assert result[0]["PlayerId"] == player_id
        assert result[0]["MLBAMID"] == mlbam_id
        # Original keys still present
        assert result[0]["playerid"] == player_id
        assert result[0][sample_key] == sample_val

    @pytest.mark.parametrize(
        ("stat_type", "detail", "rows", "sample_key", "sample_val", "player_id", "mlbam_id"),
        _STAT_TYPE_CASES,
    )
    def test_query_params(
        self,
        stat_type: str,
        detail: str,
        rows: list[dict],
        sample_key: str,
        sample_val: object,
        player_id: str,
        mlbam_id: int,
    ) -> None:
        transport = FakeTransport(_json_response([]))
        client = httpx.Client(transport=transport)
        source = FgProjectionSource(projection_type="fangraphsdc", stat_type=stat_type, client=client)

        source.fetch()

        assert transport.last_request is not None
        params = dict(transport.last_request.url.params)
        assert params["type"] == "fangraphsdc"
        assert params["stats"] == stat_type
        assert params["pos"] == "all"
        assert params["team"] == "0"
        assert params["players"] == "0"

    @pytest.mark.parametrize(
        ("stat_type", "detail", "rows", "sample_key", "sample_val", "player_id", "mlbam_id"),
        _STAT_TYPE_CASES,
    )
    def test_empty_response(
        self,
        stat_type: str,
        detail: str,
        rows: list[dict],
        sample_key: str,
        sample_val: object,
        player_id: str,
        mlbam_id: int,
    ) -> None:
        transport = FakeTransport(_json_response([]))
        client = httpx.Client(transport=transport)
        source = FgProjectionSource(projection_type="fangraphsdc", stat_type=stat_type, client=client)

        result = source.fetch()

        assert result == []

    @pytest.mark.parametrize(
        ("stat_type", "detail", "rows", "sample_key", "sample_val", "player_id", "mlbam_id"),
        _STAT_TYPE_CASES,
    )
    def test_retry_on_503(
        self,
        stat_type: str,
        detail: str,
        rows: list[dict],
        sample_key: str,
        sample_val: object,
        player_id: str,
        mlbam_id: int,
    ) -> None:
        transport = FailNTransport(fail_count=2, success_response=_json_response(rows))
        client = httpx.Client(transport=transport)
        source = FgProjectionSource(
            projection_type="fangraphsdc",
            stat_type=stat_type,
            client=client,
            retry=_NO_WAIT_RETRY,
        )

        result = source.fetch()

        assert len(result) == 1
        assert transport.call_count == 3

    @pytest.mark.parametrize(
        ("stat_type", "detail", "rows", "sample_key", "sample_val", "player_id", "mlbam_id"),
        _STAT_TYPE_CASES,
    )
    def test_exhausted_retries_raises(
        self,
        stat_type: str,
        detail: str,
        rows: list[dict],
        sample_key: str,
        sample_val: object,
        player_id: str,
        mlbam_id: int,
    ) -> None:
        transport = FailNTransport(fail_count=5, success_response=_json_response(rows))
        client = httpx.Client(transport=transport)
        source = FgProjectionSource(
            projection_type="fangraphsdc",
            stat_type=stat_type,
            client=client,
            retry=_NO_WAIT_RETRY,
        )

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch()

        assert transport.call_count == 3

    @pytest.mark.parametrize(
        ("stat_type", "detail", "rows", "sample_key", "sample_val", "player_id", "mlbam_id"),
        _STAT_TYPE_CASES,
    )
    def test_data_wrapper_response(
        self,
        stat_type: str,
        detail: str,
        rows: list[dict],
        sample_key: str,
        sample_val: object,
        player_id: str,
        mlbam_id: int,
    ) -> None:
        """API may return {"data": [...]} instead of bare array."""
        transport = FakeTransport(_json_response({"data": rows}))
        client = httpx.Client(transport=transport)
        source = FgProjectionSource(projection_type="fangraphsdc", stat_type=stat_type, client=client)

        result = source.fetch()

        assert len(result) == 1
        assert result[0]["PlayerId"] == player_id

    def test_invalid_projection_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown projection_type 'bogus'"):
            FgProjectionSource(projection_type="bogus", stat_type="bat")

    def test_invalid_stat_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stat_type 'invalid'"):
            FgProjectionSource(projection_type="fangraphsdc", stat_type="invalid")


class TestProjectionSystemsConstant:
    def test_contains_expected_systems(self) -> None:
        assert "fangraphs-dc" in PROJECTION_SYSTEMS
        assert "steamer" in PROJECTION_SYSTEMS
        assert "zips" in PROJECTION_SYSTEMS

    def test_maps_to_api_types(self) -> None:
        assert PROJECTION_SYSTEMS["fangraphs-dc"] == "fangraphsdc"
        assert PROJECTION_SYSTEMS["steamer"] == "steamer"
        assert PROJECTION_SYSTEMS["zips"] == "zips"


# --- Integration: FgProjectionSource → mapper → Loader → repo ---

_API_BATTING_ROW: dict[str, Any] = {
    "playerid": "10155",
    "xMLBAMID": 545361,
    "PlayerName": "Mike Trout",
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
    "OBP": 0.395,
    "SLG": 0.575,
    "OPS": 0.970,
    "wOBA": 0.410,
    "wRC+": 170.0,
    "WAR": 8.5,
}

_API_PITCHING_ROW: dict[str, Any] = {
    "playerid": "19755",
    "xMLBAMID": 669373,
    "PlayerName": "Corbin Burnes",
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
    "WAR": 6.0,
}


class TestFgProjectionSourceIntegration:
    def test_batting_source_to_loader_pipeline(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, fangraphs_id=10155, mlbam_id=545361)

        # Build source backed by fake transport
        transport = FakeTransport(_json_response([_API_BATTING_ROW]))
        client = httpx.Client(transport=transport)
        source = FgProjectionSource(projection_type="fangraphsdc", stat_type="bat", client=client)

        # Wire through mapper + loader
        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        players = player_repo.all()
        mapper = make_fg_projection_batting_mapper(
            players, season=2026, system="fangraphs-dc", version="2026-03-07", source_type="third_party"
        )
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(conn))
        log_repo = SqliteLoadLogRepo(SingleConnectionProvider(conn))
        loader = Loader(source, proj_repo, log_repo, mapper, "projection", provider=SingleConnectionProvider(conn))

        result = loader.load()

        assert isinstance(result, Ok)
        assert result.value.rows_loaded == 1

        results = proj_repo.get_by_player_season(player_id, 2026, system="fangraphs-dc")
        assert len(results) == 1
        assert results[0].stat_json["hr"] == 35
        assert results[0].stat_json["avg"] == 0.302
        assert results[0].version == "2026-03-07"
        assert results[0].source_type == "third_party"
        assert results[0].player_type == "batter"

    def test_pitching_source_to_loader_pipeline(self, conn: sqlite3.Connection) -> None:
        player_id = seed_player(conn, fangraphs_id=19755, mlbam_id=669373)

        transport = FakeTransport(_json_response([_API_PITCHING_ROW]))
        client = httpx.Client(transport=transport)
        source = FgProjectionSource(projection_type="steamer", stat_type="pit", client=client)

        player_repo = SqlitePlayerRepo(SingleConnectionProvider(conn))
        players = player_repo.all()
        mapper = make_fg_projection_pitching_mapper(
            players, season=2026, system="steamer", version="2026-03-07", source_type="third_party"
        )
        proj_repo = SqliteProjectionRepo(SingleConnectionProvider(conn))
        log_repo = SqliteLoadLogRepo(SingleConnectionProvider(conn))
        loader = Loader(source, proj_repo, log_repo, mapper, "projection", provider=SingleConnectionProvider(conn))

        result = loader.load()

        assert isinstance(result, Ok)
        assert result.value.rows_loaded == 1

        results = proj_repo.get_by_player_season(player_id, 2026, system="steamer")
        assert len(results) == 1
        assert results[0].stat_json["era"] == 2.80
        assert results[0].stat_json["so"] == 200
        assert results[0].version == "2026-03-07"
        assert results[0].source_type == "third_party"
        assert results[0].player_type == "pitcher"
