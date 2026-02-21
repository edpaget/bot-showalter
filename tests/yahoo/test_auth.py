import json
import time
from pathlib import Path

import httpx
import pytest

from fantasy_baseball_manager.yahoo.auth import YahooAuth, YahooTokens


class FakeTransport(httpx.BaseTransport):
    """Returns a canned response and records the last request."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return self._response


def _token_response(
    access_token: str = "new-access", refresh_token: str = "new-refresh", expires_in: int = 3600
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": expires_in,
            "token_type": "bearer",
        },
    )


class TestYahooTokensPersistence:
    def test_save_and_load_tokens(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        auth = YahooAuth("client_id", "client_secret", token_path=token_path)
        tokens = YahooTokens(access_token="access", refresh_token="refresh", expires_at=time.time() + 3600)
        auth._save_tokens(tokens)

        loaded = auth._load_tokens()
        assert loaded is not None
        assert loaded.access_token == "access"
        assert loaded.refresh_token == "refresh"

    def test_load_tokens_returns_none_when_no_file(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        auth = YahooAuth("client_id", "client_secret", token_path=token_path)
        assert auth._load_tokens() is None

    def test_load_tokens_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        token_path.write_text("not json")
        auth = YahooAuth("client_id", "client_secret", token_path=token_path)
        assert auth._load_tokens() is None


class TestYahooAuthGetAccessToken:
    def test_returns_cached_token_when_valid(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        expires_at = time.time() + 3600
        token_path.write_text(
            json.dumps({"access_token": "cached", "refresh_token": "refresh", "expires_at": expires_at})
        )
        auth = YahooAuth("client_id", "client_secret", token_path=token_path)
        assert auth.get_access_token() == "cached"

    def test_refreshes_expired_token(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        expires_at = time.time() - 100  # expired
        token_path.write_text(
            json.dumps({"access_token": "old", "refresh_token": "old-refresh", "expires_at": expires_at})
        )
        transport = FakeTransport(_token_response(access_token="refreshed"))
        client = httpx.Client(transport=transport)
        auth = YahooAuth("client_id", "client_secret", token_path=token_path, http_client=client)
        token = auth.get_access_token()
        assert token == "refreshed"

        # Verify refresh request was made correctly
        assert transport.last_request is not None
        body = transport.last_request.content.decode()
        assert "grant_type=refresh_token" in body
        assert "refresh_token=old-refresh" in body

    def test_refreshed_tokens_are_persisted(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        expires_at = time.time() - 100
        token_path.write_text(
            json.dumps({"access_token": "old", "refresh_token": "old-refresh", "expires_at": expires_at})
        )
        transport = FakeTransport(_token_response(access_token="new-access", refresh_token="new-refresh"))
        client = httpx.Client(transport=transport)
        auth = YahooAuth("client_id", "client_secret", token_path=token_path, http_client=client)
        auth.get_access_token()

        saved = json.loads(token_path.read_text())
        assert saved["access_token"] == "new-access"
        assert saved["refresh_token"] == "new-refresh"


class TestYahooAuthAuthorize:
    def test_authorize_opens_browser_and_exchanges_code(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        transport = FakeTransport(_token_response(access_token="authorized"))
        client = httpx.Client(transport=transport)

        opened_urls: list[str] = []

        def fake_opener(url: str) -> None:
            opened_urls.append(url)

        auth = YahooAuth(
            "my-client-id",
            "my-client-secret",
            token_path=token_path,
            http_client=client,
            browser_opener=fake_opener,
            code_reader=lambda: "auth-code-123",
        )
        token = auth.get_access_token()

        assert token == "authorized"
        assert len(opened_urls) == 1
        assert "my-client-id" in opened_urls[0]

        # Verify token exchange request
        assert transport.last_request is not None
        body = transport.last_request.content.decode()
        assert "grant_type=authorization_code" in body
        assert "code=auth-code-123" in body

    def test_authorize_persists_tokens(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        transport = FakeTransport(_token_response(access_token="new", refresh_token="new-ref"))
        client = httpx.Client(transport=transport)

        auth = YahooAuth(
            "id",
            "secret",
            token_path=token_path,
            http_client=client,
            browser_opener=lambda url: None,
            code_reader=lambda: "code",
        )
        auth.get_access_token()

        saved = json.loads(token_path.read_text())
        assert saved["access_token"] == "new"
        assert saved["refresh_token"] == "new-ref"


class TestYahooAuthRefresh:
    def test_refresh_sends_correct_request(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        transport = FakeTransport(_token_response())
        client = httpx.Client(transport=transport)
        auth = YahooAuth("my-id", "my-secret", token_path=token_path, http_client=client)

        tokens = auth._refresh("old-refresh-token")
        assert tokens.access_token == "new-access"
        assert tokens.refresh_token == "new-refresh"

        assert transport.last_request is not None
        body = transport.last_request.content.decode()
        assert "grant_type=refresh_token" in body
        assert "refresh_token=old-refresh-token" in body
        # Check basic auth header
        assert transport.last_request.headers.get("authorization") is not None

    def test_refresh_error_raises(self, tmp_path: Path) -> None:
        token_path = tmp_path / "yahoo_tokens.json"
        transport = FakeTransport(httpx.Response(401, json={"error": "invalid_grant"}))
        client = httpx.Client(transport=transport)
        auth = YahooAuth("id", "secret", token_path=token_path, http_client=client)

        with pytest.raises(httpx.HTTPStatusError):
            auth._refresh("bad-token")
