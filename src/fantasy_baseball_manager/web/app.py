from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import strawberry
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from strawberry.fastapi import GraphQLRouter

from fantasy_baseball_manager.config_toml import WebConfig, load_toml, load_web_config
from fantasy_baseball_manager.web.event_bus import EventBus
from fantasy_baseball_manager.web.schema import Mutation, Query, Subscription

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fantasy_baseball_manager.analysis_container import AnalysisContainer
    from fantasy_baseball_manager.domain import BreakoutPrediction, LeagueSettings, YahooLeagueInfo
    from fantasy_baseball_manager.repos import YahooTeamRepo, YahooTeamStatsRepo
    from fantasy_baseball_manager.services import KeeperPlannerService
    from fantasy_baseball_manager.web.session_manager import SessionManager
    from fantasy_baseball_manager.web.yahoo_poller_manager import YahooPollerManager


def _load_valuation_defaults() -> tuple[str, str]:
    """Load default system/version from fbm.toml [common] section."""
    toml_data = load_toml()
    common = toml_data.get("common", {})
    return common.get("system", "zar"), common.get("version", "production")


@dataclass(frozen=True)
class AppContext:
    container: AnalysisContainer
    league: LeagueSettings
    adp_provider: str
    default_system: str
    default_version: str
    web_config: WebConfig = field(default_factory=WebConfig)
    event_bus: EventBus = field(default_factory=EventBus)
    session_manager: SessionManager | None = None
    yahoo_poller_manager: YahooPollerManager | None = None
    yahoo_league_info: YahooLeagueInfo | None = None
    breakout_predictions: list[BreakoutPrediction] | None = None
    keeper_planner: KeeperPlannerService | None = None
    yahoo_team_repo: YahooTeamRepo | None = None
    yahoo_team_stats_repo: YahooTeamStatsRepo | None = None


def create_app(
    container: AnalysisContainer,
    league: LeagueSettings,
    adp_provider: str = "fantasypros",
    session_manager: SessionManager | None = None,
    event_bus: EventBus | None = None,
    yahoo_poller_manager: YahooPollerManager | None = None,
    yahoo_league_info: YahooLeagueInfo | None = None,
    breakout_predictions: list[BreakoutPrediction] | None = None,
    frontend_dir: str | None = None,
    default_system: str | None = None,
    default_version: str | None = None,
    web_config: WebConfig | None = None,
    keeper_planner: KeeperPlannerService | None = None,
    yahoo_team_repo: YahooTeamRepo | None = None,
    yahoo_team_stats_repo: YahooTeamStatsRepo | None = None,
) -> FastAPI:
    """Create a FastAPI application with a GraphQL endpoint at /graphql."""
    if default_system is None or default_version is None:
        sys_default, ver_default = _load_valuation_defaults()
        if default_system is None:
            default_system = sys_default
        if default_version is None:
            default_version = ver_default

    if web_config is None:
        web_config = load_web_config()

    app_context = AppContext(
        container=container,
        league=league,
        adp_provider=adp_provider,
        default_system=default_system,
        default_version=default_version,
        web_config=web_config,
        event_bus=event_bus or EventBus(),
        session_manager=session_manager,
        yahoo_poller_manager=yahoo_poller_manager,
        yahoo_league_info=yahoo_league_info,
        breakout_predictions=breakout_predictions,
        keeper_planner=keeper_planner,
        yahoo_team_repo=yahoo_team_repo,
        yahoo_team_stats_repo=yahoo_team_stats_repo,
    )

    schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)

    async def get_context() -> dict[str, AppContext]:
        return {"app_context": app_context}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if app_context.yahoo_poller_manager is not None:
            await app_context.yahoo_poller_manager.shutdown()

    graphql_router = GraphQLRouter(schema, context_getter=get_context)
    app = FastAPI(title="Fantasy Baseball Manager", lifespan=lifespan)
    app.include_router(graphql_router, prefix="/graphql")

    if frontend_dir is not None:
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app
