import json
import logging
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

_AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
_TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
_DEFAULT_TOKEN_PATH = Path.home() / ".config" / "fbm" / "yahoo_tokens.json"


@dataclass(frozen=True)
class YahooTokens:
    access_token: str
    refresh_token: str
    expires_at: float


class YahooAuth:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        token_path: Path = _DEFAULT_TOKEN_PATH,
        http_client: httpx.Client | None = None,
        browser_opener: Callable[[str], Any] | None = None,
        code_reader: Callable[[], str] | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_path = token_path
        self._http_client = http_client or httpx.Client(timeout=httpx.Timeout(30.0))
        self._browser_opener = browser_opener or webbrowser.open
        self._code_reader = code_reader or self._default_code_reader

    def get_access_token(self) -> str:
        """Get a valid access token, refreshing or authorizing as needed."""
        tokens = self._load_tokens()
        if tokens is not None and tokens.expires_at > time.time():
            return tokens.access_token

        if tokens is not None and tokens.refresh_token:
            logger.info("Access token expired, refreshing...")
            tokens = self._refresh(tokens.refresh_token)
            self._save_tokens(tokens)
            return tokens.access_token

        logger.info("No valid tokens found, starting authorization flow...")
        tokens = self._authorize()
        self._save_tokens(tokens)
        return tokens.access_token

    def _load_tokens(self) -> YahooTokens | None:
        if not self._token_path.exists():
            return None
        try:
            data = json.loads(self._token_path.read_text())
            return YahooTokens(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=data["expires_at"],
            )
        except json.JSONDecodeError:
            return None
        except KeyError:
            return None

    def _save_tokens(self, tokens: YahooTokens) -> None:
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        self._token_path.write_text(
            json.dumps(
                {
                    "access_token": tokens.access_token,
                    "refresh_token": tokens.refresh_token,
                    "expires_at": tokens.expires_at,
                }
            )
        )

    def _refresh(self, refresh_token: str) -> YahooTokens:
        response = self._http_client.post(
            _TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            auth=(self._client_id, self._client_secret),
        )
        response.raise_for_status()
        return self._parse_token_response(response.json())

    def _authorize(self) -> YahooTokens:
        auth_url = f"{_AUTH_URL}?client_id={self._client_id}&redirect_uri=oob&response_type=code"
        self._browser_opener(auth_url)
        code = self._code_reader()

        response = self._http_client.post(
            _TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": "oob",
            },
            auth=(self._client_id, self._client_secret),
        )
        response.raise_for_status()
        return self._parse_token_response(response.json())

    @staticmethod
    def _parse_token_response(data: dict[str, Any]) -> YahooTokens:
        return YahooTokens(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=time.time() + data["expires_in"],
        )

    @staticmethod
    def _default_code_reader() -> str:
        return input("Enter the authorization code from Yahoo: ").strip()
