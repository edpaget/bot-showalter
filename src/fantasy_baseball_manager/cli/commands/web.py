import logging
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.cli._defaults import _DataDirOpt, load_cli_defaults
from fantasy_baseball_manager.cli.factory import build_breakout_bust_report_context
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.config_yahoo import load_yahoo_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import BreakoutPrediction, LeagueSettings, Valuation, YahooLeagueInfo
from fantasy_baseball_manager.models import ModelConfig
from fantasy_baseball_manager.repos import (
    SqliteDraftSessionRepo,
    SqliteYahooLeagueRepo,
    SqliteYahooPlayerMapRepo,
    SqliteYahooTeamRepo,
)
from fantasy_baseball_manager.services import (
    KeeperPlannerService,
    PlayerEligibilityService,
    compute_adjusted_valuations,
)
from fantasy_baseball_manager.web import EventBus, SessionManager, YahooPollerManager, create_app
from fantasy_baseball_manager.yahoo.auth import YahooAuth
from fantasy_baseball_manager.yahoo.client import YahooFantasyClient
from fantasy_baseball_manager.yahoo.draft_source import YahooDraftSource
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper

logger = logging.getLogger(__name__)


def _make_valuation_adjuster(
    container: AnalysisContainer,
    league: LeagueSettings,
) -> callable:  # type: ignore[type-arg]
    """Build a ValuationAdjuster-compatible callable for keeper-adjusted draft pools."""
    eligibility = PlayerEligibilityService(
        container.position_appearance_repo,
        pitching_stats_repo=container.pitching_stats_repo,
    )

    def adjust(kept_ids: set[int], valuations: list[Valuation], season: int) -> list[Valuation]:
        projections = container.projection_repo.get_by_season(season)
        batter_positions = eligibility.get_batter_positions(season, league)
        pitcher_ids = [p.player_id for p in projections if p.player_type == "pitcher"]
        pitcher_positions = eligibility.get_pitcher_positions(season, league, pitcher_ids)
        player_ids = {v.player_id for v in valuations}
        players = container.player_repo.get_by_ids(list(player_ids))

        adjusted = compute_adjusted_valuations(
            kept_ids, projections, batter_positions, pitcher_positions, league, valuations, players
        )

        # Map AdjustedValuation back to Valuation, preserving original metadata
        orig_lookup: dict[int, Valuation] = {}
        for v in valuations:
            if v.player_id not in orig_lookup or v.value > orig_lookup[v.player_id].value:
                orig_lookup[v.player_id] = v

        result: list[Valuation] = []
        for adj in adjusted:
            orig = orig_lookup.get(adj.player_id)
            if orig is not None:
                result.append(replace(orig, value=adj.adjusted_value))
            else:
                # Player not in original valuations (shouldn't happen but be safe)
                result.append(
                    Valuation(
                        player_id=adj.player_id,
                        season=season,
                        system="zar",
                        version="1.0",
                        projection_system="",
                        projection_version="",
                        player_type=adj.player_type,
                        position=adj.position,
                        value=adj.adjusted_value,
                        rank=0,
                        category_scores={},
                    )
                )
        return result

    return adjust


def web(  # pragma: no cover
    season: Annotated[int | None, typer.Option("--season", help="Season year")] = None,
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    host: Annotated[str, typer.Option("--host", help="Server host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Server port")] = 8000,
    data_dir: _DataDirOpt = "./data",
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
    valuation_adjuster = _make_valuation_adjuster(container, league)
    session_manager = SessionManager(
        session_repo=session_repo,
        valuation_repo=container.valuation_repo,
        player_repo=container.player_repo,
        adp_repo=container.adp_repo,
        player_profile_service=container.player_profile_service,
        league=league,
        adp_provider="fantasypros",
        valuation_adjuster=valuation_adjuster,
        league_keeper_repo=container.league_keeper_repo,
        projection_repo=container.projection_repo,
    )

    # Load breakout/bust predictions if model is trained
    breakout_predictions: list[BreakoutPrediction] | None = None
    try:
        with build_breakout_bust_report_context(data_dir) as ctx:
            config = ModelConfig(seasons=[season], data_dir=data_dir)
            result = ctx.model.predict(config)  # type: ignore[union-attr]
            breakout_predictions = [
                BreakoutPrediction(
                    player_id=p["player_id"],
                    player_name=p["player_name"],
                    player_type=p["player_type"],
                    position=p["position"],
                    p_breakout=p["p_breakout"],
                    p_bust=p["p_bust"],
                    p_neutral=p["p_neutral"],
                    top_features=p.get("top_features", []),
                )
                for p in result.predictions
            ]
            logger.info("Loaded %d breakout/bust predictions", len(breakout_predictions))
    except Exception:
        logger.info("Breakout/bust model not available, skipping predictions")

    # Resolve frontend assets directory
    frontend_dir: str | None = None
    dist_path = Path(__file__).resolve().parents[4] / "frontend" / "dist"
    if dist_path.is_dir():
        frontend_dir = str(dist_path)
        logger.info("Serving frontend from %s", frontend_dir)

    # Build keeper planner if keeper cost data exists
    eligibility = PlayerEligibilityService(
        container.position_appearance_repo,
        pitching_stats_repo=container.pitching_stats_repo,
    )
    keeper_costs = container.keeper_cost_repo.find_by_season_league(season, league_name)
    valuations = container.valuation_repo.get_by_season(season, system=system, version=version)
    players_for_planner = container.player_repo.get_by_ids([v.player_id for v in valuations])
    projections_for_planner = container.projection_repo.get_by_season(season)
    batter_positions = eligibility.get_batter_positions(season, league)
    pitcher_ids = [p.player_id for p in projections_for_planner if p.player_type == "pitcher"]
    pitcher_positions = eligibility.get_pitcher_positions(season, league, pitcher_ids)
    keeper_planner: KeeperPlannerService | None = KeeperPlannerService(
        keeper_costs=keeper_costs,
        valuations=valuations,
        players=players_for_planner,
        projections=projections_for_planner,
        league=league,
        batter_positions=batter_positions,
        pitcher_positions=pitcher_positions,
    )

    yahoo_poller_manager = None
    yahoo_league_info = None
    if yahoo_config_dir is not None:
        yahoo_config = load_yahoo_config(Path(yahoo_config_dir))
        auth = YahooAuth(yahoo_config.client_id, yahoo_config.client_secret)
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

        # Build league info snapshot from DB
        league_config = yahoo_config.leagues.get(league_name)
        if league_config is not None:
            league_repo = SqliteYahooLeagueRepo(provider)
            all_leagues = league_repo.get_all()
            matched = [
                lg
                for lg in all_leagues
                if lg.season == season and lg.league_key.endswith(f".l.{league_config.league_id}")
            ]
            if matched:
                yahoo_league = matched[0]
                user_team = team_repo.get_user_team(yahoo_league.league_key)
                yahoo_league_info = YahooLeagueInfo(
                    league_key=yahoo_league.league_key,
                    league_name=yahoo_league.name,
                    season=yahoo_league.season,
                    num_teams=yahoo_league.num_teams,
                    is_keeper=yahoo_league.is_keeper,
                    max_keepers=league_config.max_keepers,
                    user_team_name=user_team.name if user_team is not None else None,
                )
                logger.info("Yahoo league: %s (%s)", yahoo_league_info.league_name, yahoo_league_info.league_key)

    app = create_app(
        container,
        league,
        session_manager=session_manager,
        event_bus=event_bus,
        yahoo_poller_manager=yahoo_poller_manager,
        yahoo_league_info=yahoo_league_info,
        breakout_predictions=breakout_predictions,
        frontend_dir=frontend_dir,
        default_system=system,
        default_version=version,
        keeper_planner=keeper_planner,
    )
    uvicorn.run(app, host=host, port=port)
