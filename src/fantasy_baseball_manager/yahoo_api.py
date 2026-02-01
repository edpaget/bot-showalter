from __future__ import annotations

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

    @property
    def oauth(self) -> Any:
        if self._oauth is None:
            self._oauth = self._oauth_factory(
                str(self._config["yahoo.client_id"]),
                str(self._config["yahoo.client_secret"]),
                store_file=str(self._config["yahoo.token_file"]),
            )
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
        return self.game.to_league(str(self._config["league.id"]))
