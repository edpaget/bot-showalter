from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from fantasy_baseball_manager.cache.factory import create_cache_store, get_cache_key
from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore


class TestCreateCacheStore:
    def test_returns_sqlite_store_with_correct_path(self, tmp_path: Path) -> None:
        config = MagicMock()
        config.__getitem__ = MagicMock(return_value=str(tmp_path / "cache.db"))

        store = create_cache_store(config)

        assert isinstance(store, SqliteCacheStore)
        assert store._db_path == tmp_path / "cache.db"

    @patch("fantasy_baseball_manager.config.create_config")
    def test_falls_back_to_create_config(self, mock_create_config: MagicMock, tmp_path: Path) -> None:
        mock_config = MagicMock()
        mock_config.__getitem__ = MagicMock(return_value=str(tmp_path / "cache.db"))
        mock_create_config.return_value = mock_config

        store = create_cache_store()

        mock_create_config.assert_called_once()
        assert isinstance(store, SqliteCacheStore)


class TestGetCacheKey:
    def test_returns_expected_key(self) -> None:
        config = MagicMock()
        config.__getitem__ = MagicMock(side_effect=lambda k: {"league.game_code": "mlb", "league.id": "123"}[k])

        result = get_cache_key(config)

        assert result == "mlb_123"

    @patch("fantasy_baseball_manager.config.create_config")
    def test_falls_back_to_create_config(self, mock_create_config: MagicMock) -> None:
        mock_config = MagicMock()
        mock_config.__getitem__ = MagicMock(side_effect=lambda k: {"league.game_code": "mlb", "league.id": "456"}[k])
        mock_create_config.return_value = mock_config

        result = get_cache_key()

        mock_create_config.assert_called_once()
        assert result == "mlb_456"
