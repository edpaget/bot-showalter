import json
from typing import Any

import httpx
import pytest

from fantasy_baseball_manager.ingest.mlb_milb_stats_source import MLBMinorLeagueBattingSource
from fantasy_baseball_manager.ingest.protocols import DataSource


class FakeTransport(httpx.BaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._response


def _make_player_stat(
    *,
    player_id: int = 545361,
    first_name: str = "Mike",
    last_name: str = "Trout",
    age: int = 24,
    team_name: str = "Syracuse Mets",
    league_name: str = "International League",
) -> dict[str, Any]:
    return {
        "player": {
            "id": player_id,
            "firstName": first_name,
            "lastName": last_name,
            "currentAge": age,
        },
        "team": {"name": team_name},
        "league": {"name": league_name},
        "stat": {
            "gamesPlayed": 120,
            "plateAppearances": 500,
            "atBats": 450,
            "hits": 130,
            "doubles": 25,
            "triples": 3,
            "homeRuns": 18,
            "runs": 70,
            "rbi": 65,
            "baseOnBalls": 40,
            "strikeOuts": 100,
            "stolenBases": 15,
            "caughtStealing": 5,
            "avg": ".289",
            "obp": ".350",
            "slg": ".480",
            "hitByPitch": 8,
            "sacFlies": 4,
            "sacBunts": 1,
        },
    }


def _fake_api_response(splits: list[dict[str, Any]]) -> httpx.Response:
    body = json.dumps({"stats": [{"splits": splits}]}).encode()
    return httpx.Response(200, content=body, headers={"content-type": "application/json"})


class TestMLBMinorLeagueBattingSource:
    def test_satisfies_datasource_protocol(self) -> None:
        source = MLBMinorLeagueBattingSource()
        assert isinstance(source, DataSource)

    def test_source_type_and_detail(self) -> None:
        source = MLBMinorLeagueBattingSource()
        assert source.source_type == "mlb_api"
        assert source.source_detail == "milb_batting"

    def test_valid_json_returns_dataframe(self) -> None:
        splits = [
            _make_player_stat(player_id=545361, first_name="Mike", last_name="Trout"),
            _make_player_stat(player_id=660271, first_name="Shohei", last_name="Ohtani"),
        ]
        response = _fake_api_response(splits)
        client = httpx.Client(transport=FakeTransport(response))
        source = MLBMinorLeagueBattingSource(client=client)

        df = source.fetch(season=2024, level="AAA")

        assert len(df) == 2
        assert df.iloc[0]["mlbam_id"] == 545361
        assert df.iloc[1]["mlbam_id"] == 660271
        assert df.iloc[0]["g"] == 120
        assert df.iloc[0]["pa"] == 500
        assert df.iloc[0]["hr"] == 18
        assert df.iloc[0]["avg"] == pytest.approx(0.289)
        assert df.iloc[0]["obp"] == pytest.approx(0.350)
        assert df.iloc[0]["slg"] == pytest.approx(0.480)
        assert df.iloc[0]["season"] == 2024
        assert df.iloc[0]["level"] == "AAA"
        assert df.iloc[0]["league"] == "International League"
        assert df.iloc[0]["team"] == "Syracuse Mets"
        assert df.iloc[0]["age"] == 24
        assert df.iloc[0]["hbp"] == 8
        assert df.iloc[0]["sf"] == 4
        assert df.iloc[0]["sh"] == 1

    def test_empty_response_returns_empty_dataframe(self) -> None:
        response = _fake_api_response([])
        client = httpx.Client(transport=FakeTransport(response))
        source = MLBMinorLeagueBattingSource(client=client)

        df = source.fetch(season=2024, level="AAA")

        assert len(df) == 0
        assert "mlbam_id" in df.columns
        assert "hr" in df.columns

    def test_http_error_raises(self) -> None:
        response = httpx.Response(500, content=b"Internal Server Error")
        client = httpx.Client(transport=FakeTransport(response))
        source = MLBMinorLeagueBattingSource(client=client)

        with pytest.raises(httpx.HTTPStatusError):
            source.fetch(season=2024, level="AAA")
