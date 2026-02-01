from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from config import config_from_dict

from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

_TEST_CONFIG_DATA: dict[str, object] = {
    "yahoo": {
        "client_id": "test_id",
        "client_secret": "test_secret",
        "token_file": "token.json",
    },
    "league": {
        "id": "12345",
        "game_code": "mlb",
        "season": 2025,
    },
}


def _make_config() -> Any:
    return config_from_dict(_TEST_CONFIG_DATA)


def test_client_accepts_config() -> None:
    cfg = _make_config()
    client = YahooFantasyClient(cfg, oauth_factory=MagicMock(), game_factory=MagicMock())
    # Construction should have no side effects — factories not called yet
    client._oauth_factory.assert_not_called()  # type: ignore[union-attr]
    client._game_factory.assert_not_called()  # type: ignore[union-attr]


def test_oauth_created_lazily() -> None:
    mock_oauth_factory = MagicMock()
    client = YahooFantasyClient(
        _make_config(),
        oauth_factory=mock_oauth_factory,
        game_factory=MagicMock(),
    )
    mock_oauth_factory.assert_not_called()

    _ = client.oauth
    mock_oauth_factory.assert_called_once_with("test_id", "test_secret", store_file="token.json")

    # Second access returns cached instance, no second call
    _ = client.oauth
    mock_oauth_factory.assert_called_once()


def test_game_created_with_oauth_and_game_code() -> None:
    mock_oauth = MagicMock()
    mock_oauth_factory = MagicMock(return_value=mock_oauth)
    mock_game_factory = MagicMock()

    client = YahooFantasyClient(
        _make_config(),
        oauth_factory=mock_oauth_factory,
        game_factory=mock_game_factory,
    )
    _ = client.game
    mock_game_factory.assert_called_once_with(mock_oauth, "mlb")


def test_get_league_delegates_to_game() -> None:
    mock_game = MagicMock()
    mock_game_factory = MagicMock(return_value=mock_game)

    client = YahooFantasyClient(
        _make_config(),
        oauth_factory=MagicMock(),
        game_factory=mock_game_factory,
    )
    league = client.get_league()
    mock_game.to_league.assert_called_once_with("12345")
    assert league is mock_game.to_league.return_value
