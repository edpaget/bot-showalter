"""Centralized service container for CLI dependency injection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.league.roster import RosterSource
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper
    from fantasy_baseball_manager.ros.protocol import ProjectionBlender


@dataclass
class ServiceConfig:
    """Configuration options for service creation."""

    no_cache: bool = False


class ServiceContainer:
    """Lazily-initialized container for CLI service dependencies.

    Supports explicit injection for testing via constructor parameters.
    When dependencies are not provided, they are created on first access
    using the default implementations.
    """

    def __init__(
        self,
        config: ServiceConfig | None = None,
        *,
        data_source: StatsDataSource | None = None,
        id_mapper: PlayerIdMapper | None = None,
        roster_source: RosterSource | None = None,
        blender: ProjectionBlender | None = None,
        yahoo_league: object | None = None,
    ) -> None:
        self._config = config or ServiceConfig()
        self._data_source = data_source
        self._id_mapper = id_mapper
        self._roster_source = roster_source
        self._blender = blender
        self._yahoo_league = yahoo_league

    @property
    def config(self) -> ServiceConfig:
        return self._config

    @cached_property
    def data_source(self) -> StatsDataSource:
        if self._data_source is not None:
            return self._data_source
        from fantasy_baseball_manager.marcel.data_source import PybaseballDataSource

        return PybaseballDataSource()

    @cached_property
    def id_mapper(self) -> PlayerIdMapper:
        if self._id_mapper is not None:
            return self._id_mapper
        from fantasy_baseball_manager.cache.factory import create_cache_store, get_cache_key
        from fantasy_baseball_manager.config import create_config
        from fantasy_baseball_manager.player_id.mapper import (
            build_cached_sfbb_mapper,
            build_sfbb_mapper,
        )

        if self._config.no_cache:
            return build_sfbb_mapper()
        config = create_config()
        ttl = int(str(config["cache.id_mappings_ttl"]))
        cache_store = create_cache_store(config)
        cache_key = get_cache_key(config)
        return build_cached_sfbb_mapper(cache_store, cache_key, ttl)

    @cached_property
    def roster_source(self) -> RosterSource:
        if self._roster_source is not None:
            return self._roster_source
        from typing import cast

        from fantasy_baseball_manager.cache.factory import create_cache_store, get_cache_key
        from fantasy_baseball_manager.cache.sources import CachedRosterSource
        from fantasy_baseball_manager.config import AppConfig, create_config
        from fantasy_baseball_manager.league.roster import YahooRosterSource
        from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

        config = create_config()
        client = YahooFantasyClient(cast("AppConfig", config))
        league = client.get_league()
        source: RosterSource = YahooRosterSource(league)
        if not self._config.no_cache:
            ttl = int(str(config["cache.rosters_ttl"]))
            cache_store = create_cache_store(config)
            cache_key = get_cache_key(config)
            source = CachedRosterSource(source, cache_store, cache_key, ttl)
        return source

    @cached_property
    def blender(self) -> ProjectionBlender:
        if self._blender is not None:
            return self._blender
        from fantasy_baseball_manager.ros.blender import BayesianBlender

        return BayesianBlender()

    @cached_property
    def yahoo_league(self) -> object:
        if self._yahoo_league is not None:
            return self._yahoo_league
        from typing import cast

        from fantasy_baseball_manager.config import AppConfig, create_config
        from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

        config = create_config()
        client = YahooFantasyClient(cast("AppConfig", config))
        return client.get_league()


_container: ServiceContainer | None = None


def get_container() -> ServiceContainer:
    """Get the global service container, creating one if needed."""
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def set_container(container: ServiceContainer | None) -> None:
    """Set or reset the global service container.

    Pass None to reset, which will cause get_container() to create
    a fresh container on next access.
    """
    global _container
    _container = container
