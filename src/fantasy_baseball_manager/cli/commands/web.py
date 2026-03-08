from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.repos import SqliteDraftSessionRepo
from fantasy_baseball_manager.web import SessionManager, create_app


def web(  # pragma: no cover
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    host: Annotated[str, typer.Option("--host", help="Server host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Server port")] = 8000,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Start the GraphQL API server."""
    league = load_league(league_name, Path.cwd())
    conn = create_connection(Path(data_dir) / "fbm.db")
    provider = SingleConnectionProvider(conn)
    container = AnalysisContainer(provider)

    session_repo = SqliteDraftSessionRepo(provider)
    session_manager = SessionManager(
        session_repo=session_repo,
        valuation_repo=container.valuation_repo,
        player_repo=container.player_repo,
        adp_repo=container.adp_repo,
        player_profile_service=container.player_profile_service,
        league=league,
        adp_provider="fantasypros",
    )

    app = create_app(container, league, session_manager=session_manager)
    uvicorn.run(app, host=host, port=port)
