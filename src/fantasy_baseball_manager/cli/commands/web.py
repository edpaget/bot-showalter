from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.cli._defaults import load_cli_defaults
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.config_yahoo import load_yahoo_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.repos import SqliteDraftSessionRepo, SqliteYahooPlayerMapRepo, SqliteYahooTeamRepo
from fantasy_baseball_manager.web import EventBus, SessionManager, YahooPollerManager, create_app
from fantasy_baseball_manager.yahoo.auth import YahooAuth
from fantasy_baseball_manager.yahoo.client import YahooFantasyClient
from fantasy_baseball_manager.yahoo.draft_source import YahooDraftSource
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper


def web(  # pragma: no cover
    season: Annotated[int | None, typer.Option("--season", help="Season year")] = None,
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    host: Annotated[str, typer.Option("--host", help="Server host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Server port")] = 8000,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
    yahoo_config_dir: Annotated[str | None, typer.Option("--yahoo-config-dir", help="Yahoo config directory")] = None,
) -> None:
    """Start the GraphQL API server."""
    defaults = load_cli_defaults()
    if season is None:
        season = defaults.season
    if system is None:
        system = defaults.system
    if version is None:
        version = defaults.version
    league = load_league(league_name, Path.cwd())
    conn = create_connection(Path(data_dir) / "fbm.db")
    provider = SingleConnectionProvider(conn)
    container = AnalysisContainer(provider)

    event_bus = EventBus()

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

    yahoo_poller_manager = None
    if yahoo_config_dir is not None:
        config = load_yahoo_config(Path(yahoo_config_dir))
        auth = YahooAuth(config.client_id, config.client_secret)
        client = YahooFantasyClient(auth)
        player_map_repo = SqliteYahooPlayerMapRepo(provider)
        player_mapper = YahooPlayerMapper(player_map_repo, container.player_repo)
        draft_source = YahooDraftSource(client, player_mapper)
        team_repo = SqliteYahooTeamRepo(provider)

        yahoo_poller_manager = YahooPollerManager(
            _draft_source=draft_source,
            _session_manager=session_manager,
            _event_bus=event_bus,
            _team_repo=team_repo,
        )

    app = create_app(
        container,
        league,
        session_manager=session_manager,
        event_bus=event_bus,
        yahoo_poller_manager=yahoo_poller_manager,
    )
    uvicorn.run(app, host=host, port=port)
