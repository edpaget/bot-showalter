import datetime
import functools
import logging
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from fantasy_baseball_manager.cli._output import console, print_error, print_keeper_decisions
from fantasy_baseball_manager.cli.factory import YahooContext, build_yahoo_context
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.config_yahoo import YahooConfigError, load_yahoo_config, resolve_default_league
from fantasy_baseball_manager.domain import (
    DraftBoard,
    DraftBoardRow,
    Err,
    LeagueSettings,
    Ok,
    YahooDraftPick,
    YahooLeague,
    YahooPlayerMap,
    YahooTeam,
)
from fantasy_baseball_manager.services import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftSession,
    build_draft_board,
    build_draft_roster_slots,
    build_keeper_histories,
    build_team_map,
    compute_surplus,
    derive_keeper_costs,
    ingest_yahoo_pick,
    recommend,
    set_keeper_cost,
)
from fantasy_baseball_manager.services import (
    draft_report as compute_draft_report,
)
from fantasy_baseball_manager.yahoo.auth import YahooAuth
from fantasy_baseball_manager.yahoo.draft_poller import YahooDraftPoller
from fantasy_baseball_manager.yahoo.draft_source import YahooDraftSource
from fantasy_baseball_manager.yahoo.league_source import YahooLeagueSource
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper
from fantasy_baseball_manager.yahoo.roster_source import YahooRosterSource

logger = logging.getLogger(__name__)

yahoo_app = typer.Typer(name="yahoo", help="Yahoo Fantasy integration")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


# ---------------------------------------------------------------------------
# Yahoo draft setup helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YahooDraftSetup:
    engine: DraftEngine
    board: DraftBoard
    team_map: dict[str, int]
    source: YahooDraftSource
    replayed_count: int


def _build_yahoo_draft_setup(
    ctx: YahooContext,
    league_key: str,
    season: int,
    fbm_league: LeagueSettings,
    system: str,
    version: str,
    provider: str,
) -> YahooDraftSetup:
    teams = ctx.yahoo_team_repo.get_by_league_key(league_key)
    if not teams:
        msg = f"No teams found for league key '{league_key}'. Run 'fbm yahoo sync' first."
        raise ValueError(msg)

    team_map = build_team_map(teams)

    user_team = ctx.yahoo_team_repo.get_user_team(league_key)
    if user_team is None:
        msg = "No user team found. Run 'fbm yahoo sync' first."
        raise ValueError(msg)

    valuations = ctx.valuation_repo.get_by_season(season, system=system)
    valuations = [v for v in valuations if v.version == version]

    player_ids = [v.player_id for v in valuations]
    players_list = ctx.player_repo.get_by_ids(player_ids)
    player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players_list if p.id is not None}

    adp_list = ctx.adp_repo.get_by_season(season, provider=provider)

    board = build_draft_board(valuations, fbm_league, player_names, adp=adp_list if adp_list else None)
    draft_players: list[DraftBoardRow] = board.rows

    yahoo_league = ctx.yahoo_league_repo.get_by_league_key(league_key)
    if yahoo_league is None:
        msg = "League metadata not found. Run 'fbm yahoo sync' first."
        raise ValueError(msg)

    roster_slots = build_draft_roster_slots(fbm_league)
    draft_type = yahoo_league.draft_type
    draft_format = DraftFormat.AUCTION if "auction" in draft_type.lower() else DraftFormat.SNAKE
    draft_config = DraftConfig(
        teams=yahoo_league.num_teams,
        roster_slots=roster_slots,
        format=draft_format,
        user_team=user_team.team_id,
        season=season,
        budget=260 if draft_format == DraftFormat.AUCTION else 0,
    )

    engine = DraftEngine()
    engine.start(draft_players, draft_config)

    mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
    source = YahooDraftSource(ctx.client, mapper)

    existing_picks = source.fetch_draft_results(league_key, season)
    for pick in existing_picks:
        ctx.yahoo_draft_repo.upsert(pick)
        ingest_yahoo_pick(engine.pick, set(engine.state.available_pool), pick, team_map)

    ctx.conn.commit()

    return YahooDraftSetup(
        engine=engine,
        board=board,
        team_map=team_map,
        source=source,
        replayed_count=len(existing_picks),
    )


@yahoo_app.command("auth")
def yahoo_auth(  # pragma: no cover
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Authenticate with Yahoo Fantasy API (triggers OAuth flow)."""
    try:
        config = load_yahoo_config(Path(config_dir))
    except YahooConfigError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    auth = YahooAuth(config.client_id, config.client_secret)
    try:
        token = auth.get_access_token()
    except Exception as exc:
        print_error(f"Authentication failed: {exc}")
        raise typer.Exit(code=1) from None

    console.print(f"[bold green]Authenticated.[/bold green] Token: {token[:8]}...")


@yahoo_app.command("sync")
def yahoo_sync(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Sync league metadata from Yahoo Fantasy API."""
    try:
        config = load_yahoo_config(Path(config_dir))
    except YahooConfigError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    if league is None:
        try:
            league = resolve_default_league(config)
        except YahooConfigError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

    if league not in config.leagues:
        print_error(f"League '{league}' not found in [yahoo.leagues]")
        raise typer.Exit(code=1)

    league_config = config.leagues[league]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        # Get game key for current season
        game_key = ctx.client.get_game_key(2026)
        league_key = f"{game_key}.l.{league_config.league_id}"

        source = YahooLeagueSource(ctx.client)
        result = source.fetch(league_key=league_key, game_key=game_key)

        # Upsert league
        league_data = result["league"]
        yahoo_league = YahooLeague(
            league_key=league_data["league_key"],
            name=league_data["name"],
            season=league_data["season"],
            num_teams=league_data["num_teams"],
            draft_type=league_data["draft_type"],
            is_keeper=league_data["is_keeper"],
            game_key=league_data["game_key"],
        )
        ctx.yahoo_league_repo.upsert(yahoo_league)

        # Upsert teams
        for team_data in result["teams"]:
            yahoo_team = YahooTeam(
                team_key=team_data["team_key"],
                league_key=team_data["league_key"],
                team_id=team_data["team_id"],
                name=team_data["name"],
                manager_name=team_data["manager_name"],
                is_owned_by_user=team_data["is_owned_by_user"],
            )
            ctx.yahoo_team_repo.upsert(yahoo_team)

        ctx.conn.commit()

    console.print(f"[bold green]Synced[/bold green] {yahoo_league.name} ({yahoo_league.league_key})")
    console.print(f"  Season: {yahoo_league.season}")
    console.print(f"  Teams: {yahoo_league.num_teams}")
    console.print(f"  Draft type: {yahoo_league.draft_type}")
    console.print(f"  Keeper: {'yes' if yahoo_league.is_keeper else 'no'}")
    for team_data in result["teams"]:
        marker = " (you)" if team_data["is_owned_by_user"] else ""
        console.print(f"  - {team_data['name']} ({team_data['manager_name']}){marker}")


@yahoo_app.command("map-player")
def yahoo_map_player(  # pragma: no cover
    yahoo_key: Annotated[str, typer.Argument(help="Yahoo player key (e.g. 449.p.12345)")],
    player_name: Annotated[str, typer.Argument(help="Player name (e.g. 'Mike Trout')")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")] = "batter",
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Manually map a Yahoo player key to an internal player."""
    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        # Search for the player by name
        parts = player_name.split()
        if len(parts) < 2:
            print_error("Player name must include first and last name")
            raise typer.Exit(code=1)

        candidates = ctx.player_repo.get_by_last_name(parts[-1])
        first = parts[0].lower()
        matches = [p for p in candidates if p.name_first and p.name_first.lower() == first]

        if not matches:
            # Fall back to search_by_name
            matches = ctx.player_repo.search_by_name(player_name)

        if not matches:
            print_error(f"No player found matching '{player_name}'")
            raise typer.Exit(code=1)

        if len(matches) > 1:
            console.print(f"[yellow]Multiple matches for '{player_name}':[/yellow]")
            for m in matches:
                console.print(f"  id={m.id} {m.name_first} {m.name_last} (mlbam={m.mlbam_id})")
            print_error("Ambiguous match. Use a more specific name.")
            raise typer.Exit(code=1)

        player = matches[0]
        assert player.id is not None  # noqa: S101 - type narrowing

        mapping = YahooPlayerMap(
            yahoo_player_key=yahoo_key,
            player_id=player.id,
            player_type=player_type,
            yahoo_name=player_name,
            yahoo_team="",
            yahoo_positions="",
        )
        ctx.yahoo_player_map_repo.upsert(mapping)
        ctx.conn.commit()

    console.print(
        f"[bold green]Mapped[/bold green] {yahoo_key} -> "
        f"{player.name_first} {player.name_last} (id={player.id}, type={player_type})"
    )


@yahoo_app.command("rosters")
def yahoo_rosters(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Fetch and display all teams' rosters from Yahoo Fantasy."""
    try:
        config = load_yahoo_config(Path(config_dir))
    except YahooConfigError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    if league is None:
        try:
            league = resolve_default_league(config)
        except YahooConfigError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

    if league not in config.leagues:
        print_error(f"League '{league}' not found in [yahoo.leagues]")
        raise typer.Exit(code=1)

    league_config = config.leagues[league]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        game_key = ctx.client.get_game_key(2026)
        league_key = f"{game_key}.l.{league_config.league_id}"

        teams = ctx.yahoo_team_repo.get_by_league_key(league_key)
        if not teams:
            print_error(f"No teams found for league '{league}'. Run 'fbm yahoo sync' first.")
            raise typer.Exit(code=1)

        mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
        source = YahooRosterSource(ctx.client, mapper)
        today = datetime.date.today()

        for team in teams:
            roster = source.fetch_team_roster(
                team_key=team.team_key,
                league_key=league_key,
                season=2026,
                week=1,
                as_of=today,
            )
            ctx.yahoo_roster_repo.save_snapshot(roster)

            marker = " (you)" if team.is_owned_by_user else ""
            console.print(f"\n[bold]{team.name}[/bold]{marker}")

            table = Table(show_header=True)
            table.add_column("Pos")
            table.add_column("Player")
            table.add_column("Status")
            table.add_column("Acquired")
            table.add_column("Mapped")

            for entry in roster.entries:
                mapped = "[green]yes[/green]" if entry.player_id is not None else "[red]no[/red]"
                table.add_row(
                    entry.position,
                    entry.player_name,
                    entry.roster_status,
                    entry.acquisition_type,
                    mapped,
                )
            console.print(table)

        ctx.conn.commit()

    console.print(f"\n[bold green]Fetched rosters for {len(teams)} teams.[/bold green]")


@yahoo_app.command("my-roster")
def yahoo_my_roster(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Fetch your roster and show projections/valuations."""
    try:
        config = load_yahoo_config(Path(config_dir))
    except YahooConfigError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    if league is None:
        try:
            league = resolve_default_league(config)
        except YahooConfigError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

    if league not in config.leagues:
        print_error(f"League '{league}' not found in [yahoo.leagues]")
        raise typer.Exit(code=1)

    league_config = config.leagues[league]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        game_key = ctx.client.get_game_key(2026)
        league_key = f"{game_key}.l.{league_config.league_id}"

        user_team = ctx.yahoo_team_repo.get_user_team(league_key)
        if user_team is None:
            print_error("No user team found. Run 'fbm yahoo sync' first.")
            raise typer.Exit(code=1)

        mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
        source = YahooRosterSource(ctx.client, mapper)
        today = datetime.date.today()

        roster = source.fetch_team_roster(
            team_key=user_team.team_key,
            league_key=league_key,
            season=season,
            week=1,
            as_of=today,
        )
        ctx.yahoo_roster_repo.save_snapshot(roster)

        # Build projection/valuation lookup
        projections = ctx.projection_repo.get_by_season(season)
        proj_by_player: dict[int, Any] = {}
        for proj in projections:
            if proj.player_id not in proj_by_player:
                proj_by_player[proj.player_id] = proj

        valuations = ctx.valuation_repo.get_by_season(season)
        val_by_player: dict[int, float] = {}
        for val in valuations:
            if val.player_id not in val_by_player:
                val_by_player[val.player_id] = val.value

        ctx.conn.commit()

    console.print(f"\n[bold]{user_team.name}[/bold] — {league}")

    table = Table(show_header=True)
    table.add_column("Pos")
    table.add_column("Player")
    table.add_column("Status")
    table.add_column("Type")
    table.add_column("Value", justify="right")
    table.add_column("System")

    for entry in roster.entries:
        value_str = ""
        system_str = ""
        if entry.player_id is not None:
            val = val_by_player.get(entry.player_id)
            if val is not None:
                value_str = f"{val:.1f}"
            proj = proj_by_player.get(entry.player_id)
            if proj is not None:
                system_str = f"{proj.system}/{proj.version}"

        table.add_row(
            entry.position,
            entry.player_name,
            entry.roster_status,
            entry.acquisition_type,
            value_str,
            system_str,
        )
    console.print(table)


@yahoo_app.command("draft-results")
def yahoo_draft_results(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Fetch and display draft results from Yahoo Fantasy."""
    try:
        config = load_yahoo_config(Path(config_dir))
    except YahooConfigError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    if league is None:
        try:
            league = resolve_default_league(config)
        except YahooConfigError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

    if league not in config.leagues:
        print_error(f"League '{league}' not found in [yahoo.leagues]")
        raise typer.Exit(code=1)

    league_config = config.leagues[league]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        game_key = ctx.client.get_game_key(season)
        league_key = f"{game_key}.l.{league_config.league_id}"

        mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
        source = YahooDraftSource(ctx.client, mapper)

        picks = source.fetch_draft_results(league_key, season)

        for pick in picks:
            ctx.yahoo_draft_repo.upsert(pick)

        ctx.conn.commit()

    if not picks:
        console.print("[yellow]No draft results found.[/yellow]")
        return

    table = Table(title=f"Draft Results — {league} ({season})", show_header=True)
    table.add_column("Round", justify="right")
    table.add_column("Pick", justify="right")
    table.add_column("Team")
    table.add_column("Player")
    table.add_column("Position")
    table.add_column("Cost", justify="right")

    for pick in picks:
        cost_str = f"${pick.cost}" if pick.cost is not None else ""
        table.add_row(
            str(pick.round),
            str(pick.pick),
            pick.team_key.split(".")[-1],
            pick.player_name,
            pick.position,
            cost_str,
        )
    console.print(table)
    console.print(f"\n[bold green]Fetched {len(picks)} draft picks.[/bold green]")


@yahoo_app.command("draft-live")
def yahoo_draft_live(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    league_name: Annotated[str, typer.Option("--league-config", help="League name from fbm.toml")] = "default",
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    poll_interval: Annotated[float, typer.Option("--poll-interval", help="Poll interval in seconds")] = 5.0,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Start a live Yahoo draft session with auto-pick ingestion."""
    try:
        config = load_yahoo_config(Path(config_dir))
    except YahooConfigError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    if league is None:
        try:
            league = resolve_default_league(config)
        except YahooConfigError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

    if league not in config.leagues:
        print_error(f"League '{league}' not found in [yahoo.leagues]")
        raise typer.Exit(code=1)

    league_config = config.leagues[league]
    fbm_league = load_league(league_name, Path.cwd())

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        game_key = ctx.client.get_game_key(season)
        league_key = f"{game_key}.l.{league_config.league_id}"

        try:
            setup = _build_yahoo_draft_setup(ctx, league_key, season, fbm_league, system, version, provider)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

    if setup.replayed_count:
        console.print(f"[green]Replayed {setup.replayed_count} existing picks.[/green]")

    pick_queue: queue.Queue[YahooDraftPick] = queue.Queue()
    poller = YahooDraftPoller(
        source=setup.source,
        league_key=league_key,
        season=season,
        interval=poll_interval,
        pick_queue=pick_queue,
    )

    report_fn = functools.partial(
        compute_draft_report,
        batting_categories=setup.board.batting_categories,
        pitching_categories=setup.board.pitching_categories,
    )

    session = DraftSession(
        engine=setup.engine,
        players=setup.board.rows,
        console=console,
        recommend_fn=recommend,
        report_fn=report_fn,
        yahoo_pick_queue=pick_queue,
        team_map=setup.team_map,
    )

    console.print(f"[bold]Starting live Yahoo draft[/bold] — polling every {poll_interval}s")
    poller.start()
    try:
        session.run()
    finally:
        poller.stop()


# ---------------------------------------------------------------------------
# Helpers for keeper commands
# ---------------------------------------------------------------------------


def _resolve_league_context(league: str | None, config_dir: str) -> tuple[str, Any]:
    """Resolve Yahoo config and league name. Returns (league_name, config)."""
    config = load_yahoo_config(Path(config_dir))

    if league is None:
        league = resolve_default_league(config)

    if league not in config.leagues:
        print_error(f"League '{league}' not found in [yahoo.leagues]")
        raise typer.Exit(code=1)

    return league, config


def _derive_and_store_keeper_costs(
    ctx: YahooContext,
    league_key: str,
    season: int,
    league_name: str,
    cost_floor: float,
) -> None:
    """Derive keeper costs from Yahoo draft/roster and upsert, respecting manual overrides."""
    prior_season = season - 1
    prior_game_key = ctx.client.get_game_key(prior_season)
    prior_league_key = f"{prior_game_key}.l.{league_key.split('.l.')[1]}"

    mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
    draft_source = YahooDraftSource(ctx.client, mapper)
    roster_source = YahooRosterSource(ctx.client, mapper)

    draft_picks = draft_source.fetch_draft_results(prior_league_key, prior_season)

    user_team = ctx.yahoo_team_repo.get_user_team(league_key)
    if user_team is None:
        print_error("No user team found. Run 'fbm yahoo sync' first.")
        raise typer.Exit(code=1)

    today = datetime.date.today()
    roster = roster_source.fetch_team_roster(
        team_key=user_team.team_key,
        league_key=league_key,
        season=season,
        week=1,
        as_of=today,
    )

    derived = derive_keeper_costs(draft_picks, list(roster.entries), league_name, season, cost_floor)

    # Respect manual overrides: skip upserting players with non-yahoo_* source
    existing = ctx.keeper_repo.find_by_season_league(season, league_name)
    manual_player_ids = {kc.player_id for kc in existing if not kc.source.startswith("yahoo_")}
    to_upsert = [kc for kc in derived if kc.player_id not in manual_player_ids]

    if to_upsert:
        ctx.keeper_repo.upsert_batch(to_upsert)
        ctx.conn.commit()


# ---------------------------------------------------------------------------
# Keeper commands
# ---------------------------------------------------------------------------


@yahoo_app.command("keeper-costs")
def yahoo_keeper_costs(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    cost_floor: Annotated[float, typer.Option("--cost-floor", help="Minimum keeper cost for FA pickups")] = 1.0,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Derive keeper costs from Yahoo draft history and current roster."""
    league_name, config = _resolve_league_context(league, config_dir)
    league_config = config.leagues[league_name]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        game_key = ctx.client.get_game_key(season)
        league_key = f"{game_key}.l.{league_config.league_id}"

        _derive_and_store_keeper_costs(ctx, league_key, season, league_name, cost_floor)

        # Display results
        all_costs = ctx.keeper_repo.find_by_season_league(season, league_name)
        player_ids = [kc.player_id for kc in all_costs]
        players_list = ctx.player_repo.get_by_ids(player_ids)

    if not all_costs:
        console.print("[yellow]No keeper costs derived.[/yellow]")
        return

    player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players_list if p.id is not None}

    table = Table(title=f"Keeper Costs — {league_name} ({season})", show_header=True)
    table.add_column("Player", justify="left")
    table.add_column("Cost", justify="right")
    table.add_column("Source", justify="left")

    for kc in sorted(all_costs, key=lambda c: c.cost, reverse=True):
        name = player_names.get(kc.player_id, f"ID:{kc.player_id}")
        table.add_row(name, f"${kc.cost:.0f}", kc.source)

    console.print(table)
    console.print(f"\n[bold green]Derived {len(all_costs)} keeper costs.[/bold green]")


@yahoo_app.command("keeper-decisions")
def yahoo_keeper_decisions(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    threshold: Annotated[float, typer.Option("--threshold", help="Minimum surplus for keep recommendation")] = 0.0,
    decay: Annotated[float, typer.Option("--decay", help="Decay factor for multi-year surplus")] = 0.85,
    cost_floor: Annotated[float, typer.Option("--cost-floor", help="Minimum keeper cost for FA pickups")] = 1.0,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Show keeper decisions ranked by surplus value using Yahoo-derived costs."""
    league_name, config = _resolve_league_context(league, config_dir)
    league_config = config.leagues[league_name]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        game_key = ctx.client.get_game_key(season)
        league_key = f"{game_key}.l.{league_config.league_id}"

        # Auto-derive costs first
        _derive_and_store_keeper_costs(ctx, league_key, season, league_name, cost_floor)

        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league_name)
        if not keeper_costs:
            console.print("[yellow]No keeper costs found.[/yellow]")
            return

        valuations = ctx.valuation_repo.get_by_season(season, system)
        players = ctx.player_repo.all()

    decisions = compute_surplus(keeper_costs, valuations, players, threshold=threshold, decay=decay)
    print_keeper_decisions(decisions)


@yahoo_app.command("keeper-history")
def yahoo_keeper_history(  # pragma: no cover
    player: Annotated[str, typer.Argument(help="Player name to look up")],
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Show a player's keeper cost history across seasons."""
    league_name, _config = _resolve_league_context(league, config_dir)

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        matches = ctx.player_repo.search_by_name(player)
        if not matches:
            print_error(f"No player found matching '{player}'")
            raise typer.Exit(code=1)
        if len(matches) > 1:
            console.print(f"[yellow]Multiple matches for '{player}':[/yellow]")
            for m in matches:
                console.print(f"  id={m.id} {m.name_first} {m.name_last}")
            print_error("Ambiguous match. Use a more specific name.")
            raise typer.Exit(code=1)

        resolved = matches[0]
        assert resolved.id is not None  # noqa: S101 - type narrowing

        all_costs = ctx.keeper_repo.find_by_player(resolved.id)
        histories = build_keeper_histories(all_costs, [resolved], league_name)

    if not histories:
        console.print(
            f"[yellow]No keeper history for {resolved.name_first} {resolved.name_last} in '{league_name}'.[/yellow]"
        )
        return

    history = histories[0]
    table = Table(title=f"Keeper History — {history.player_name} ({league_name})", show_header=True)
    table.add_column("Season", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Source", justify="left")

    for entry in history.entries:
        table.add_row(str(entry.season), f"${entry.cost:.0f}", entry.source)

    console.print(table)


@yahoo_app.command("keeper-cost-set")
def yahoo_keeper_cost_set(  # pragma: no cover
    player_name: Annotated[str, typer.Argument(help="Player name")],
    cost: Annotated[float, typer.Option("--cost", help="Keeper cost")],
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    years: Annotated[int, typer.Option("--years", help="Years remaining on contract")] = 1,
    source: Annotated[str, typer.Option("--source", help="Cost source type")] = "manual",
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Manually set a keeper cost for a player in a Yahoo league."""
    league_name, _config = _resolve_league_context(league, config_dir)

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        result = set_keeper_cost(
            player_name, cost, season, league_name, ctx.player_repo, ctx.keeper_repo, years, source
        )
        match result:
            case Ok(kc):
                ctx.conn.commit()
                player = ctx.player_repo.get_by_id(kc.player_id)
                name = f"{player.name_first} {player.name_last}" if player else "Unknown"
                console.print(f"[bold green]Set keeper cost[/bold green] for {name}: ${cost:.0f} ({source}, {years}yr)")
            case Err(msg):
                print_error(str(msg))
                raise typer.Exit(code=1)
