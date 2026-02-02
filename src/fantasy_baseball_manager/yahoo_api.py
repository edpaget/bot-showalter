from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yahoo_fantasy_api import Game, League
from yahoo_oauth import OAuth2

if TYPE_CHECKING:
    from collections.abc import Callable

    from fantasy_baseball_manager.config import AppConfig


class YahooFantasyClient:
    """Thin wrapper around yahoo-fantasy-api with dependency injection."""

    def __init__(
        self,
        config: AppConfig,
        *,
        oauth_factory: Callable[..., Any] | None = None,
        game_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._oauth_factory = oauth_factory or OAuth2
        self._game_factory = game_factory or Game
        self._oauth: Any | None = None
        self._game: Any | None = None

    def _ensure_credentials_file(self) -> str:
        """Ensure the credentials file exists with current consumer key/secret.

        Creates the parent directory and file if missing. Updates consumer_key
        and consumer_secret if they differ from config, preserving any existing
        token data.

        Returns the resolved absolute path to the credentials file.
        """
        creds_path = Path(str(self._config["yahoo.credentials_file"])).expanduser()
        consumer_key = str(self._config["yahoo.client_id"])
        consumer_secret = str(self._config["yahoo.client_secret"])

        if creds_path.exists():
            data: dict[str, Any] = json.loads(creds_path.read_text())
            if data.get("consumer_key") == consumer_key and data.get("consumer_secret") == consumer_secret:
                return str(creds_path)
            data["consumer_key"] = consumer_key
            data["consumer_secret"] = consumer_secret
        else:
            creds_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "consumer_key": consumer_key,
                "consumer_secret": consumer_secret,
            }

        creds_path.write_text(json.dumps(data))
        return str(creds_path)

    @property
    def oauth(self) -> Any:
        if self._oauth is None:
            creds_file = self._ensure_credentials_file()
            self._oauth = self._oauth_factory(None, None, from_file=creds_file)
        return self._oauth

    @property
    def game(self) -> Any:
        if self._game is None:
            self._game = self._game_factory(
                self.oauth,
                str(self._config["league.game_code"]),
            )
        return self._game

    def get_league(self) -> League:
        season = str(self._config["league.season"])
        league_id = str(self._config["league.id"])
        league_keys = self.game.league_ids(seasons=[season])
        suffix = f".l.{league_id}"
        matching = [k for k in league_keys if k.endswith(suffix)]
        if not matching:
            msg = f"No league with id {league_id} found for season {season}"
            raise ValueError(msg)
        return self.game.to_league(matching[0])
