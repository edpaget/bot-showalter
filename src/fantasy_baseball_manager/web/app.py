from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

from fantasy_baseball_manager.web.event_bus import EventBus
from fantasy_baseball_manager.web.schema import Mutation, Query, Subscription

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fantasy_baseball_manager.analysis_container import AnalysisContainer
    from fantasy_baseball_manager.domain import LeagueSettings
    from fantasy_baseball_manager.web.session_manager import SessionManager
    from fantasy_baseball_manager.web.yahoo_poller_manager import YahooPollerManager


@dataclass(frozen=True)
class AppContext:
    container: AnalysisContainer
    league: LeagueSettings
    adp_provider: str
    event_bus: EventBus = field(default_factory=EventBus)
    session_manager: SessionManager | None = None
    yahoo_poller_manager: YahooPollerManager | None = None


def create_app(
    container: AnalysisContainer,
    league: LeagueSettings,
    adp_provider: str = "fantasypros",
    session_manager: SessionManager | None = None,
    event_bus: EventBus | None = None,
    yahoo_poller_manager: YahooPollerManager | None = None,
) -> FastAPI:
    """Create a FastAPI application with a GraphQL endpoint at /graphql."""
    app_context = AppContext(
        container=container,
        league=league,
        adp_provider=adp_provider,
        event_bus=event_bus or EventBus(),
        session_manager=session_manager,
        yahoo_poller_manager=yahoo_poller_manager,
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

    return app
