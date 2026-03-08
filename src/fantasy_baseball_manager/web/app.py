from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

from fantasy_baseball_manager.web.schema import Query

if TYPE_CHECKING:
    from fantasy_baseball_manager.analysis_container import AnalysisContainer
    from fantasy_baseball_manager.domain import LeagueSettings


@dataclass(frozen=True)
class AppContext:
    container: AnalysisContainer
    league: LeagueSettings
    adp_provider: str


def create_app(
    container: AnalysisContainer,
    league: LeagueSettings,
    adp_provider: str = "fantasypros",
) -> FastAPI:
    """Create a FastAPI application with a GraphQL endpoint at /graphql."""
    app_context = AppContext(container=container, league=league, adp_provider=adp_provider)

    schema = strawberry.Schema(query=Query)

    async def get_context() -> dict[str, AppContext]:
        return {"app_context": app_context}

    graphql_router = GraphQLRouter(schema, context_getter=get_context)
    app = FastAPI(title="Fantasy Baseball Manager")
    app.include_router(graphql_router, prefix="/graphql")

    return app
