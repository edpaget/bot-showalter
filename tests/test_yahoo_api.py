from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

from config import config_from_dict

from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

_TEST_CONFIG_DATA: dict[str, object] = {
    "yahoo": {
        "client_id": "test_id",
        "client_secret": "test_secret",
        "credentials_file": "~/.config/fbm/credentials.json",
    },
    "league": {
        "id": "12345",
        "game_code": "mlb",
        "season": 2025,
    },
}


def _make_config(overrides: dict[str, object] | None = None) -> Any:
    data = _TEST_CONFIG_DATA
    if overrides:
        data = {**_TEST_CONFIG_DATA, **overrides}
    return config_from_dict(data)


def test_client_accepts_config() -> None:
    cfg = _make_config()
    client = YahooFantasyClient(cfg, oauth_factory=MagicMock(), game_factory=MagicMock())
    # Construction should have no side effects — factories not called yet
    client._oauth_factory.assert_not_called()  # type: ignore[union-attr]
    client._game_factory.assert_not_called()  # type: ignore[union-attr]


def test_oauth_created_lazily_with_from_file(tmp_path: Path) -> None:
    creds_file = tmp_path / "credentials.json"
    cfg = _make_config({"yahoo": {
        "client_id": "test_id",
        "client_secret": "test_secret",
        "credentials_file": str(creds_file),
    }})
    mock_oauth_factory = MagicMock()
    client = YahooFantasyClient(
        cfg,
        oauth_factory=mock_oauth_factory,
        game_factory=MagicMock(),
    )
    mock_oauth_factory.assert_not_called()

    _ = client.oauth
    mock_oauth_factory.assert_called_once_with(None, None, from_file=str(creds_file))

    # Credentials file should have been created with consumer_key/secret
    data = json.loads(creds_file.read_text())
    assert data["consumer_key"] == "test_id"
    assert data["consumer_secret"] == "test_secret"

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


class TestEnsureCredentialsFile:
    def test_creates_directory_and_file_when_missing(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "subdir" / "credentials.json"
        cfg = _make_config({"yahoo": {
            "client_id": "my_id",
            "client_secret": "my_secret",
            "credentials_file": str(creds_file),
        }})
        client = YahooFantasyClient(cfg, oauth_factory=MagicMock(), game_factory=MagicMock())

        result = client._ensure_credentials_file()

        assert result == str(creds_file)
        assert creds_file.exists()
        data = json.loads(creds_file.read_text())
        assert data == {"consumer_key": "my_id", "consumer_secret": "my_secret"}

    def test_preserves_existing_token_data(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "credentials.json"
        existing_data = {
            "consumer_key": "my_id",
            "consumer_secret": "my_secret",
            "access_token": "abc123",
            "token_type": "bearer",
        }
        creds_file.write_text(json.dumps(existing_data))

        cfg = _make_config({"yahoo": {
            "client_id": "my_id",
            "client_secret": "my_secret",
            "credentials_file": str(creds_file),
        }})
        client = YahooFantasyClient(cfg, oauth_factory=MagicMock(), game_factory=MagicMock())

        client._ensure_credentials_file()

        data = json.loads(creds_file.read_text())
        assert data["access_token"] == "abc123"
        assert data["token_type"] == "bearer"
        assert data["consumer_key"] == "my_id"
        assert data["consumer_secret"] == "my_secret"

    def test_updates_consumer_credentials_if_changed(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "credentials.json"
        existing_data = {
            "consumer_key": "old_id",
            "consumer_secret": "old_secret",
            "access_token": "abc123",
        }
        creds_file.write_text(json.dumps(existing_data))

        cfg = _make_config({"yahoo": {
            "client_id": "new_id",
            "client_secret": "new_secret",
            "credentials_file": str(creds_file),
        }})
        client = YahooFantasyClient(cfg, oauth_factory=MagicMock(), game_factory=MagicMock())

        client._ensure_credentials_file()

        data = json.loads(creds_file.read_text())
        assert data["consumer_key"] == "new_id"
        assert data["consumer_secret"] == "new_secret"
        assert data["access_token"] == "abc123"

    def test_no_write_when_credentials_unchanged(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "credentials.json"
        existing_data = {
            "consumer_key": "my_id",
            "consumer_secret": "my_secret",
            "access_token": "abc123",
        }
        creds_file.write_text(json.dumps(existing_data))
        original_mtime = creds_file.stat().st_mtime

        cfg = _make_config({"yahoo": {
            "client_id": "my_id",
            "client_secret": "my_secret",
            "credentials_file": str(creds_file),
        }})
        client = YahooFantasyClient(cfg, oauth_factory=MagicMock(), game_factory=MagicMock())

        client._ensure_credentials_file()

        assert creds_file.stat().st_mtime == original_mtime
