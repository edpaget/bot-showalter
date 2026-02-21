import httpx
import pytest
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_none

from fantasy_baseball_manager.yahoo.client import YahooFantasyClient

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
    def __init__(self, fail_count: int, success_response: httpx.Response) -> None:
        self._fail_count = fail_count
        self._success_response = success_response
        self._call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise httpx.TransportError("connection failed")
        return self._success_response

    @property
    def call_count(self) -> int:
        return self._call_count


class FakeAuth:
    """Fake YahooAuth that returns a fixed token without any HTTP calls."""

    def __init__(self, token: str = "test-token") -> None:
        self._token = token

    def get_access_token(self) -> str:
        return self._token


_LEAGUE_SETTINGS_RESPONSE = {
    "fantasy_content": {
        "league": [
            {
                "league_key": "449.l.12345",
                "league_id": "12345",
                "name": "Test League",
                "season": "2026",
                "num_teams": 12,
                "draft_status": "postdraft",
                "league_type": "private",
                "renew": "422.l.54321",
                "game_code": "mlb",
                "is_finished": 0,
            },
            {
                "settings": [
                    {
                        "draft_type": "live_standard_draft",
                        "uses_keeper": "1",
                    }
                ]
            },
        ]
    }
}

_TEAMS_RESPONSE = {
    "fantasy_content": {
        "league": [
            {"league_key": "449.l.12345"},
            {
                "teams": {
                    "0": {
                        "team": [
                            [
                                {"team_key": "449.l.12345.t.1"},
                                {"team_id": "1"},
                                {"name": "Team One"},
                                [],
                                [],
                                [],
                                {"managers": [{"manager": {"nickname": "Alice", "is_current_login": "1"}}]},
                            ]
                        ]
                    },
                    "1": {
                        "team": [
                            [
                                {"team_key": "449.l.12345.t.2"},
                                {"team_id": "2"},
                                {"name": "Team Two"},
                                [],
                                [],
                                [],
                                {"managers": [{"manager": {"nickname": "Bob"}}]},
                            ]
                        ]
                    },
                    "count": 2,
                }
            },
        ]
    }
}

_GAME_KEY_RESPONSE = {
    "fantasy_content": {
        "games": {
            "0": {
                "game": [
                    {
                        "game_key": "449",
                        "game_id": "449",
                        "name": "Baseball",
                        "code": "mlb",
                        "season": "2026",
                    }
                ]
            },
            "count": 1,
        }
    }
}


class TestYahooFantasyClientGetLeagueSettings:
    def test_sends_correct_request(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_LEAGUE_SETTINGS_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client)  # type: ignore[arg-type]
        yahoo_client.get_league_settings("449.l.12345")

        assert transport.last_request is not None
        url = str(transport.last_request.url)
        assert "league/449.l.12345/settings" in url
        assert "format=json" in url

    def test_includes_auth_header(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_LEAGUE_SETTINGS_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(token="my-token"), client=client)  # type: ignore[arg-type]
        yahoo_client.get_league_settings("449.l.12345")

        assert transport.last_request is not None
        assert transport.last_request.headers["authorization"] == "Bearer my-token"

    def test_returns_parsed_response(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_LEAGUE_SETTINGS_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client)  # type: ignore[arg-type]
        result = yahoo_client.get_league_settings("449.l.12345")
        assert "fantasy_content" in result


class TestYahooFantasyClientGetTeams:
    def test_sends_correct_request(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_TEAMS_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client)  # type: ignore[arg-type]
        yahoo_client.get_teams("449.l.12345")

        assert transport.last_request is not None
        url = str(transport.last_request.url)
        assert "league/449.l.12345/teams" in url
        assert "format=json" in url

    def test_returns_parsed_response(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_TEAMS_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client)  # type: ignore[arg-type]
        result = yahoo_client.get_teams("449.l.12345")
        assert "fantasy_content" in result


class TestYahooFantasyClientGetGameKey:
    def test_sends_correct_request(self) -> None:
        transport = FakeTransport(httpx.Response(200, json=_GAME_KEY_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client)  # type: ignore[arg-type]
        result = yahoo_client.get_game_key(2026)

        assert transport.last_request is not None
        url = str(transport.last_request.url)
        assert "games;game_codes=mlb;seasons=2026" in url
        assert result == "449"


class TestYahooFantasyClientRetry:
    def test_retries_on_transport_error(self) -> None:
        transport = FailNTransport(2, httpx.Response(200, json=_LEAGUE_SETTINGS_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client, retry=_NO_WAIT_RETRY)  # type: ignore[arg-type]
        result = yahoo_client.get_league_settings("449.l.12345")
        assert "fantasy_content" in result
        assert transport.call_count == 3

    def test_raises_after_max_retries(self) -> None:
        transport = FailNTransport(5, httpx.Response(200, json=_LEAGUE_SETTINGS_RESPONSE))
        client = httpx.Client(transport=transport)
        yahoo_client = YahooFantasyClient(auth=FakeAuth(), client=client, retry=_NO_WAIT_RETRY)  # type: ignore[arg-type]
        with pytest.raises(httpx.TransportError):
            yahoo_client.get_league_settings("449.l.12345")
