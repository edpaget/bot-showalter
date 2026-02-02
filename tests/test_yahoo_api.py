from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

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
    cfg = _make_config(
        {
            "yahoo": {
                "client_id": "test_id",
                "client_secret": "test_secret",
                "credentials_file": str(creds_file),
            }
        }
    )
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


def test_get_league_uses_season_from_config() -> None:
    mock_game = MagicMock()
    mock_game.league_ids.return_value = ["469.l.12345", "469.l.99999"]
    mock_game_factory = MagicMock(return_value=mock_game)

    client = YahooFantasyClient(
        _make_config(),
        oauth_factory=MagicMock(),
        game_factory=mock_game_factory,
    )
    league = client.get_league()
    mock_game.league_ids.assert_called_once_with(seasons=["2025"])
    mock_game.to_league.assert_called_once_with("469.l.12345")
    assert league is mock_game.to_league.return_value


def test_get_league_raises_when_no_matching_league() -> None:
    mock_game = MagicMock()
    mock_game.league_ids.return_value = ["469.l.99999"]
    mock_game_factory = MagicMock(return_value=mock_game)

    client = YahooFantasyClient(
        _make_config(),
        oauth_factory=MagicMock(),
        game_factory=mock_game_factory,
    )

    with pytest.raises(ValueError, match="No league with id 12345 found for season 2025"):
        client.get_league()


class TestGetLeagueForSeason:
    def _make_client(self, mock_game: MagicMock) -> YahooFantasyClient:
        cfg = _make_config({"league": {"id": "12345", "game_code": "mlb", "season": 2026}})
        return YahooFantasyClient(
            cfg,
            oauth_factory=MagicMock(),
            game_factory=MagicMock(return_value=mock_game),
        )

    def test_walks_one_step_back(self) -> None:
        mock_game = MagicMock()
        # Current league (2026)
        mock_game.league_ids.return_value = ["470.l.12345"]
        current_league = MagicMock()
        current_league.settings.return_value = {"renew": "469_54321"}

        target_league = MagicMock()
        target_league.settings.return_value = {"season": 2025}

        mock_game.to_league.side_effect = [current_league, target_league]

        client = self._make_client(mock_game)
        result = client.get_league_for_season(2025)

        assert result is target_league
        mock_game.to_league.assert_any_call("469.l.54321")

    def test_walks_multiple_steps_back(self) -> None:
        mock_game = MagicMock()
        mock_game.league_ids.return_value = ["470.l.12345"]

        current_league = MagicMock()
        current_league.settings.return_value = {"renew": "469_54321"}

        mid_league = MagicMock()
        mid_league.settings.return_value = {"renew": "468_11111"}

        target_league = MagicMock()
        target_league.settings.return_value = {"season": 2024}

        mock_game.to_league.side_effect = [current_league, mid_league, target_league]

        client = self._make_client(mock_game)
        result = client.get_league_for_season(2024)

        assert result is target_league
        mock_game.to_league.assert_any_call("468.l.11111")

    def test_raises_when_renew_empty(self) -> None:
        mock_game = MagicMock()
        mock_game.league_ids.return_value = ["470.l.12345"]

        current_league = MagicMock()
        current_league.settings.return_value = {"renew": ""}

        mock_game.to_league.return_value = current_league

        client = self._make_client(mock_game)
        with pytest.raises(ValueError, match=r"Cannot walk renewal chain.*2024"):
            client.get_league_for_season(2024)

    def test_raises_when_target_ahead_of_current(self) -> None:
        mock_game = MagicMock()
        mock_game.league_ids.return_value = ["470.l.12345"]

        client = self._make_client(mock_game)
        with pytest.raises(ValueError, match=r"Target season 2027.*ahead"):
            client.get_league_for_season(2027)

    def test_returns_current_league_when_same_season(self) -> None:
        mock_game = MagicMock()
        mock_game.league_ids.return_value = ["470.l.12345"]

        client = self._make_client(mock_game)
        result = client.get_league_for_season(2026)

        assert result is mock_game.to_league.return_value
        mock_game.to_league.assert_called_once_with("470.l.12345")


class TestEnsureCredentialsFile:
    def test_creates_directory_and_file_when_missing(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "subdir" / "credentials.json"
        cfg = _make_config(
            {
                "yahoo": {
                    "client_id": "my_id",
                    "client_secret": "my_secret",
                    "credentials_file": str(creds_file),
                }
            }
        )
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

        cfg = _make_config(
            {
                "yahoo": {
                    "client_id": "my_id",
                    "client_secret": "my_secret",
                    "credentials_file": str(creds_file),
                }
            }
        )
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

        cfg = _make_config(
            {
                "yahoo": {
                    "client_id": "new_id",
                    "client_secret": "new_secret",
                    "credentials_file": str(creds_file),
                }
            }
        )
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

        cfg = _make_config(
            {
                "yahoo": {
                    "client_id": "my_id",
                    "client_secret": "my_secret",
                    "credentials_file": str(creds_file),
                }
            }
        )
        client = YahooFantasyClient(cfg, oauth_factory=MagicMock(), game_factory=MagicMock())

        client._ensure_credentials_file()

        assert creds_file.stat().st_mtime == original_mtime
