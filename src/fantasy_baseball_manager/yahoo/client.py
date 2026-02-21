import logging
from collections.abc import Callable
from typing import Any

import httpx

from fantasy_baseball_manager.ingest._retry import default_http_retry
from fantasy_baseball_manager.yahoo.auth import YahooAuth

logger = logging.getLogger(__name__)

_BASE_URL = "https://fantasysports.yahooapis.com/fantasy/v2/"
_DEFAULT_RETRY = default_http_retry("yahoo_api")


class YahooFantasyClient:
    def __init__(
        self,
        auth: YahooAuth,
        client: httpx.Client | None = None,
        retry: Callable[..., Callable[..., Any]] = _DEFAULT_RETRY,
    ) -> None:
        self._auth = auth
        self._client = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._get_with_retry = retry(self._do_get)

    def get_league_settings(self, league_key: str) -> dict[str, Any]:
        url = f"{_BASE_URL}league/{league_key}/settings"
        return self._get_with_retry(url)

    def get_teams(self, league_key: str) -> dict[str, Any]:
        url = f"{_BASE_URL}league/{league_key}/teams"
        return self._get_with_retry(url)

    def get_game_key(self, season: int) -> str:
        url = f"{_BASE_URL}games;game_codes=mlb;seasons={season}"
        data = self._get_with_retry(url)
        games = data["fantasy_content"]["games"]
        game = games["0"]["game"][0]
        return game["game_key"]

    def _do_get(self, url: str) -> dict[str, Any]:
        token = self._auth.get_access_token()
        response = self._client.get(
            url,
            params={"format": "json"},
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()
