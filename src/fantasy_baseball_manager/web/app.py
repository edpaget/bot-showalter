from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

from fantasy_baseball_manager.web.schema import Mutation, Query

if TYPE_CHECKING:
    from fantasy_baseball_manager.analysis_container import AnalysisContainer
    from fantasy_baseball_manager.domain import LeagueSettings
    from fantasy_baseball_manager.web.session_manager import SessionManager


@dataclass(frozen=True)
class AppContext:
    container: AnalysisContainer
    league: LeagueSettings
    adp_provider: str
    session_manager: SessionManager | None = None


def create_app(
    container: AnalysisContainer,
    league: LeagueSettings,
    adp_provider: str = "fantasypros",
    session_manager: SessionManager | None = None,
) -> FastAPI:
    """Create a FastAPI application with a GraphQL endpoint at /graphql."""
    app_context = AppContext(
        container=container,
        league=league,
        adp_provider=adp_provider,
        session_manager=session_manager,
    )

    schema = strawberry.Schema(query=Query, mutation=Mutation)

    async def get_context() -> dict[str, AppContext]:
        return {"app_context": app_context}

    graphql_router = GraphQLRouter(schema, context_getter=get_context)
    app = FastAPI(title="Fantasy Baseball Manager")
    app.include_router(graphql_router, prefix="/graphql")

    return app
