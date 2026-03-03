import httpx

from fantasy_baseball_manager.yahoo.client import YahooFantasyClient


class FakeAuth:
    def get_access_token(self) -> str:
        return "test-token"


class FakeTransport(httpx.BaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return self._response


_MULTIPLE_SEASONS_RESPONSE = {
    "fantasy_content": {
        "games": {
            "0": {"game": [{"game_key": "449", "season": "2026"}]},
            "1": {"game": [{"game_key": "422", "season": "2025"}]},
            "2": {"game": [{"game_key": "404", "season": "2024"}]},
            "count": 3,
        }
    }
}

_SINGLE_SEASON_RESPONSE = {
    "fantasy_content": {
        "games": {
            "0": {"game": [{"game_key": "449", "season": "2026"}]},
            "count": 1,
        }
    }
}

_EMPTY_RESPONSE = {
    "fantasy_content": {
        "games": {
            "count": 0,
        }
    }
}


class TestGetAvailableSeasons:
    def test_multiple_seasons_sorted_descending(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_MULTIPLE_SEASONS_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client)  # type: ignore[arg-type]

        result = yahoo_client.get_available_seasons()

        assert result == [("449", 2026), ("422", 2025), ("404", 2024)]

    def test_sends_correct_request(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_MULTIPLE_SEASONS_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client)  # type: ignore[arg-type]
        yahoo_client.get_available_seasons()

        assert transport.last_request is not None
        url = str(transport.last_request.url)
        assert "games;game_codes=mlb" in url

    def test_empty_seasons(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_EMPTY_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client)  # type: ignore[arg-type]

        result = yahoo_client.get_available_seasons()

        assert result == []

    def test_single_season(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_SINGLE_SEASON_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client)  # type: ignore[arg-type]

        result = yahoo_client.get_available_seasons()

        assert result == [("449", 2026)]
