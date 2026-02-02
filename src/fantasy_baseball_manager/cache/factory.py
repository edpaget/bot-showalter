from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore

if TYPE_CHECKING:
    from fantasy_baseball_manager.config import AppConfig


def create_cache_store(config: AppConfig | None = None) -> SqliteCacheStore:
    """Build a SqliteCacheStore from the app config's ``cache.db_path``."""
    if config is None:
        from fantasy_baseball_manager.config import create_config

        config = create_config()
    db_path = Path(str(config["cache.db_path"])).expanduser()
    return SqliteCacheStore(db_path)


def get_cache_key(config: AppConfig | None = None) -> str:
    """Return ``"{game_code}_{season}_{league_id}"`` derived from the app config."""
    if config is None:
        from fantasy_baseball_manager.config import create_config

        config = create_config()
    return f"{config['league.game_code']}_{config['league.season']}_{config['league.id']}"
