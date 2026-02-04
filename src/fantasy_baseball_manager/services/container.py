"""Centralized service container for CLI dependency injection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import ConfigurationSet

    from fantasy_baseball_manager.cache.sqlite_store import SqliteCacheStore
    from fantasy_baseball_manager.config import AppConfig
    from fantasy_baseball_manager.league.roster import RosterSource
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.pipeline.skill_data import SkillDataSource
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper
    from fantasy_baseball_manager.ros.protocol import ProjectionBlender

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Configuration options for service creation.

    Attributes:
        no_cache: Disable caching for all services.
        league_id: Override the league ID from config file.
        season: Override the season from config file.
    """

    no_cache: bool = False
    league_id: str | None = None
    season: int | None = None


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
        skill_data_source: SkillDataSource | None = None,
    ) -> None:
        self._config = config or ServiceConfig()
        self._data_source = data_source
        self._id_mapper = id_mapper
        self._roster_source = roster_source
        self._blender = blender
        self._yahoo_league = yahoo_league
        self._skill_data_source = skill_data_source

    @property
    def config(self) -> ServiceConfig:
        return self._config

    @cached_property
    def app_config(self) -> ConfigurationSet:
        """Application config with league_id/season overrides applied."""
        from fantasy_baseball_manager.config import create_config

        return create_config(
            league_id=self._config.league_id,
            season=self._config.season,
        )

    @cached_property
    def cache_store(self) -> SqliteCacheStore:
        """Cache store for persisting data."""
        from fantasy_baseball_manager.cache.factory import create_cache_store

        return create_cache_store(self.app_config)

    @cached_property
    def cache_key(self) -> str:
        """Cache key derived from league configuration."""
        from fantasy_baseball_manager.cache.factory import get_cache_key

        return get_cache_key(self.app_config)

    def invalidate_caches(self, namespaces: tuple[str, ...] = ("rosters", "sfbb_csv")) -> None:
        """Invalidate cached data for given namespaces."""
        for ns in namespaces:
            self.cache_store.invalidate(ns, self.cache_key)
        logger.debug("Invalidated caches %s for key=%s", namespaces, self.cache_key)

    def _create_app_config(self) -> object:
        """Create an AppConfig with league_id/season overrides applied.

        Deprecated: Use app_config property instead.
        """
        return self.app_config

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
        from fantasy_baseball_manager.player_id.mapper import (
            build_cached_sfbb_mapper,
            build_sfbb_mapper,
        )

        if self._config.no_cache:
            return build_sfbb_mapper()
        ttl = int(str(self.app_config["cache.id_mappings_ttl"]))
        return build_cached_sfbb_mapper(self.cache_store, self.cache_key, ttl)

    @cached_property
    def _roster_source_and_league(self) -> tuple[RosterSource, object]:
        """Build roster source and league together (cached)."""
        if self._roster_source is not None:
            # If roster_source was injected, use yahoo_league or a placeholder
            league = self._yahoo_league if self._yahoo_league is not None else object()
            return self._roster_source, league

        from typing import cast

        from fantasy_baseball_manager.cache.sources import CachedRosterSource
        from fantasy_baseball_manager.league.roster import YahooRosterSource
        from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

        config = self.app_config
        client = YahooFantasyClient(cast("AppConfig", config))

        # For keeper leagues in predraft, use previous season's rosters
        target_season: int | None = None
        if config["league.is_keeper"]:
            current_league = client.get_league()
            draft_status = current_league.settings().get("draft_status", "")
            logger.debug("Keeper league draft_status=%r", draft_status)
            if draft_status == "predraft":
                target_season = int(str(config["league.season"])) - 1
                logger.debug("Using previous season %d for roster source", target_season)

        league = client.get_league_for_season(target_season) if target_season is not None else client.get_league()
        source: RosterSource = YahooRosterSource(league)
        if not self._config.no_cache:
            ttl = int(str(config["cache.rosters_ttl"]))
            source = CachedRosterSource(source, self.cache_store, self.cache_key, ttl)
        return source, league

    @property
    def roster_source(self) -> RosterSource:
        """Roster source for fetching team rosters."""
        return self._roster_source_and_league[0]

    @property
    def roster_league(self) -> object:
        """Yahoo league object associated with the roster source.

        For keeper leagues in predraft, this is the previous season's league.
        """
        return self._roster_source_and_league[1]

    @cached_property
    def blender(self) -> ProjectionBlender:
        if self._blender is not None:
            return self._blender
        from fantasy_baseball_manager.ros.blender import BayesianBlender

        return BayesianBlender()

    @cached_property
    def skill_data_source(self) -> SkillDataSource:
        """Skill data source for year-over-year skill change detection."""
        if self._skill_data_source is not None:
            return self._skill_data_source
        from fantasy_baseball_manager.pipeline.skill_data import (
            CachedSkillDataSource,
            CompositeSkillDataSource,
            FanGraphsSkillDataSource,
            StatcastSprintSpeedSource,
        )

        fangraphs = FanGraphsSkillDataSource()
        sprint = StatcastSprintSpeedSource()
        composite = CompositeSkillDataSource(fangraphs, sprint, self.id_mapper)

        if self._config.no_cache:
            return composite

        return CachedSkillDataSource(composite, self.cache_store)

    @cached_property
    def yahoo_league(self) -> object:
        if self._yahoo_league is not None:
            return self._yahoo_league
        from typing import cast

        from fantasy_baseball_manager.yahoo_api import YahooFantasyClient

        config = self._create_app_config()
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
