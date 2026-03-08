import datetime
import functools
import logging
import queue
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from fantasy_baseball_manager.cli._output import (
    console,
    print_error,
    print_keeper_decisions,
    print_keeper_draft_needs,
    print_league_keeper_overview,
)
from fantasy_baseball_manager.cli.factory import YahooContext, build_yahoo_context
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.config_yahoo import (
    YahooConfig,
    YahooConfigError,
    load_yahoo_config,
    resolve_default_league,
)
from fantasy_baseball_manager.db.pool import ConnectionPool, SingleConnectionProvider
from fantasy_baseball_manager.domain import (
    Err,
    Ok,
    YahooDraftPick,
    YahooPlayerMap,
)
from fantasy_baseball_manager.name_utils import resolve_players
from fantasy_baseball_manager.repos import (
    SqlitePitchingStatsRepo,
    SqlitePlayerRepo,
    SqlitePositionAppearanceRepo,
    SqliteYahooPlayerMapRepo,
)
from fantasy_baseball_manager.services import (
    DraftSession,
    PlayerEligibilityService,
    adjust_valuations_for_league_keepers,
    build_keeper_draft_needs,
    build_keeper_histories,
    build_league_keeper_overview,
    build_yahoo_draft_setup,
    compute_surplus,
    derive_and_store_keeper_costs,
    derive_best_n_keeper_costs,
    ensure_prior_season_teams,
    fetch_league_rosters,
    recommend,
    set_keeper_cost,
    sync_league_metadata,
    sync_transactions,
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
from fantasy_baseball_manager.yahoo.transaction_source import YahooTransactionSource

logger = logging.getLogger(__name__)

yahoo_app = typer.Typer(name="yahoo", help="Yahoo Fantasy integration")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


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
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
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
        league_key = resolve_league_key(ctx, league_config.league_id, season)
        game_key = ctx.client.get_game_key(season)

        league_source = YahooLeagueSource(ctx.client)
        yahoo_league = sync_league_metadata(
            league_source=league_source,
            league_repo=ctx.yahoo_league_repo,
            team_repo=ctx.yahoo_team_repo,
            league_key=league_key,
            game_key=game_key,
            is_keeper=league_config.keeper,
        )
        ctx.conn.commit()
        teams = ctx.yahoo_team_repo.get_by_league_key(league_key)

    console.print(f"[bold green]Synced[/bold green] {yahoo_league.name} ({yahoo_league.league_key})")
    console.print(f"  Season: {yahoo_league.season}")
    console.print(f"  Teams: {yahoo_league.num_teams}")
    console.print(f"  Draft type: {yahoo_league.draft_type}")
    console.print(f"  Keeper: {'yes' if yahoo_league.is_keeper else 'no'}")
    for team in teams:
        marker = " (you)" if team.is_owned_by_user else ""
        console.print(f"  - {team.name} ({team.manager_name}){marker}")


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
            # Fall back to resolve_players for accent/nickname handling
            matches = resolve_players(ctx.player_repo, player_name)

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
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
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
        league_key = resolve_league_key(ctx, league_config.league_id, season)
        game_key = ctx.client.get_game_key(season)

        teams = ctx.yahoo_team_repo.get_by_league_key(league_key)
        if not teams:
            # Auto-sync metadata for this league key
            console.print("[yellow]No teams found. Running sync...[/yellow]")
            league_source = YahooLeagueSource(ctx.client)
            sync_league_metadata(
                league_source=league_source,
                league_repo=ctx.yahoo_league_repo,
                team_repo=ctx.yahoo_team_repo,
                league_key=league_key,
                game_key=game_key,
            )
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
                season=season,
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
        league_key = resolve_league_key(ctx, league_config.league_id, season)

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
        league_key = resolve_league_key(ctx, league_config.league_id, season)

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
        league_key = resolve_league_key(ctx, league_config.league_id, season)

        mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
        draft_source = YahooDraftSource(ctx.client, mapper)
        yahoo_client = ctx.client
        try:
            setup = build_yahoo_draft_setup(
                team_repo=ctx.yahoo_team_repo,
                league_repo=ctx.yahoo_league_repo,
                valuation_repo=ctx.valuation_repo,
                player_repo=ctx.player_repo,
                adp_repo=ctx.adp_repo,
                draft_repo=ctx.yahoo_draft_repo,
                draft_source=draft_source,
                league_key=league_key,
                season=season,
                fbm_league=fbm_league,
                system=system,
                version=version,
                provider=provider,
            )
            ctx.conn.commit()
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

    if setup.replayed_count:
        console.print(f"[green]Replayed {setup.replayed_count} existing picks.[/green]")

    # Create a pool-backed source for the background poller thread.
    # The main thread's connection (from build_yahoo_context) is closed after
    # the with block exits, and SQLite rejects cross-thread access by default.
    # A ConnectionPool with check_same_thread=False solves both issues.
    poller_pool = ConnectionPool(Path(data_dir) / "fbm.db", size=1)
    poller_mapper = YahooPlayerMapper(
        SqliteYahooPlayerMapRepo(poller_pool),
        SqlitePlayerRepo(poller_pool),
    )
    poller_source = YahooDraftSource(yahoo_client, poller_mapper)

    pick_queue: queue.Queue[YahooDraftPick] = queue.Queue()
    poller = YahooDraftPoller(
        source=poller_source,
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
        id_aliases=setup.id_aliases,
    )

    console.print(f"[bold]Starting live Yahoo draft[/bold] — polling every {poll_interval}s")
    poller.start()
    try:
        session.run()
    finally:
        poller.stop()
        poller_pool.close_all()


@yahoo_app.command("transactions")
def yahoo_transactions(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    days: Annotated[int, typer.Option("--days", help="Show transactions from last N days")] = 7,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Fetch and display recent league transactions."""
    league_name, config = _resolve_league_context(league, config_dir)
    league_config = config.leagues[league_name]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        league_key = resolve_league_key(ctx, league_config.league_id, season)

        mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
        txn_source = YahooTransactionSource(ctx.client, mapper)
        sync_transactions(
            transaction_source=txn_source,
            transaction_repo=ctx.yahoo_transaction_repo,
            league_key=league_key,
        )
        ctx.conn.commit()

        # Display recent
        recent = ctx.yahoo_transaction_repo.get_recent(league_key, days=days)

    if not recent:
        console.print(f"[yellow]No transactions in the last {days} days.[/yellow]")
        return

    table = Table(title=f"Transactions — {league_name} (last {days} days)", show_header=True)
    table.add_column("Date")
    table.add_column("Type")
    table.add_column("Status")
    table.add_column("Players")
    table.add_column("Team")

    for txn, players in recent:
        date_str = txn.timestamp.strftime("%Y-%m-%d %H:%M")
        player_strs: list[str] = []
        for p in players:
            prefix = "+" if p.type == "add" else "-"
            player_strs.append(f"{prefix} {p.player_name}")
        players_str = ", ".join(player_strs) if player_strs else ""
        team_key_short = txn.trader_team_key.split(".")[-1] if txn.trader_team_key else ""
        table.add_row(date_str, txn.type, txn.status, players_str, team_key_short)

    console.print(table)
    console.print(f"\n[bold green]Showing {len(recent)} transactions.[/bold green]")


@yahoo_app.command("refresh")
def yahoo_refresh(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Incrementally sync all league data (rosters + transactions)."""
    league_name, config = _resolve_league_context(league, config_dir)
    league_config = config.leagues[league_name]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        league_key = resolve_league_key(ctx, league_config.league_id, season)
        game_key = ctx.client.get_game_key(season)

        # Step 1: Ensure teams exist (auto-sync if needed)
        teams = ctx.yahoo_team_repo.get_by_league_key(league_key)
        if not teams:
            console.print("[yellow]No teams found. Running sync first...[/yellow]")
            league_source = YahooLeagueSource(ctx.client)
            sync_league_metadata(
                league_source=league_source,
                league_repo=ctx.yahoo_league_repo,
                team_repo=ctx.yahoo_team_repo,
                league_key=league_key,
                game_key=game_key,
            )
            teams = ctx.yahoo_team_repo.get_by_league_key(league_key)

        # Step 2: Sync rosters
        mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
        roster_source = YahooRosterSource(ctx.client, mapper)
        today = datetime.date.today()

        roster_count = 0
        for team in teams:
            roster = roster_source.fetch_team_roster(
                team_key=team.team_key,
                league_key=league_key,
                season=season,
                week=1,
                as_of=today,
            )
            ctx.yahoo_roster_repo.save_snapshot(roster)
            roster_count += 1

        # Step 3: Sync transactions incrementally
        txn_source = YahooTransactionSource(ctx.client, mapper)
        txn_count = sync_transactions(
            transaction_source=txn_source,
            transaction_repo=ctx.yahoo_transaction_repo,
            league_key=league_key,
        )

        ctx.conn.commit()

    console.print(f"[bold green]Refresh complete[/bold green] — {league_name}")
    console.print(f"  Rosters synced: {roster_count}")
    console.print(f"  New transactions: {txn_count}")


# ---------------------------------------------------------------------------
# Helpers for keeper commands
# ---------------------------------------------------------------------------


def _resolve_league_context(league: str | None, config_dir: str) -> tuple[str, YahooConfig]:
    """Resolve Yahoo config and league name. Returns (league_name, config)."""
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

    return league, config


def resolve_league_key(ctx: YahooContext, league_id: int, season: int, *, current_year: int | None = None) -> str:
    """Resolve the Yahoo league key for any season.

    For the current (or future) season, constructs directly from league_id.
    For prior seasons, follows the renew chain from the current-season league.
    """
    if current_year is None:
        current_year = datetime.date.today().year

    game_key = ctx.client.get_game_key(season)

    if season >= current_year:
        return f"{game_key}.l.{league_id}"

    # For prior seasons, resolve via renew chain from current league
    current_game_key = ctx.client.get_game_key(current_year)
    current_league_key = f"{current_game_key}.l.{league_id}"
    return _resolve_prior_league_key(ctx, current_league_key, season)


def _resolve_prior_league_key(ctx: YahooContext, league_key: str, prior_season: int) -> str:
    """Resolve the prior-season league key using the stored renew chain.

    Falls back to same-league-ID construction with a warning if renew is unavailable.
    """
    stored_league = ctx.yahoo_league_repo.get_by_league_key(league_key)
    if stored_league is not None and stored_league.renew is not None:
        # Yahoo returns renew in underscore format (e.g. "458_135575");
        # convert to league-key format ("458.l.135575").
        return stored_league.renew.replace("_", ".l.", 1)

    prior_game_key = ctx.client.get_game_key(prior_season)
    league_id = league_key.split(".l.")[1]
    fallback = f"{prior_game_key}.l.{league_id}"
    logger.warning("No renew chain for %s — assuming prior league key %s", league_key, fallback)
    return fallback


def _ensure_prior_season(
    ctx: YahooContext,
    prior_league_key: str,
    prior_game_key: str,
) -> None:
    """Auto-sync prior-season league/team metadata if missing, then commit."""
    league_source = YahooLeagueSource(ctx.client)
    ensure_prior_season_teams(
        team_repo=ctx.yahoo_team_repo,
        league_source=league_source,
        league_repo=ctx.yahoo_league_repo,
        prior_league_key=prior_league_key,
        prior_game_key=prior_game_key,
    )
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

        prior_season = season - 1
        prior_league_key = _resolve_prior_league_key(ctx, league_key, prior_season)
        prior_game_key = prior_league_key.split(".l.")[0]

        try:
            _ensure_prior_season(ctx, prior_league_key, prior_game_key)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

        mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
        roster_source = YahooRosterSource(ctx.client, mapper)
        try:
            if league_config.keeper_format == "best_n":
                derive_best_n_keeper_costs(
                    roster_source=roster_source,
                    team_repo=ctx.yahoo_team_repo,
                    keeper_repo=ctx.keeper_repo,
                    prior_league_key=prior_league_key,
                    prior_season=prior_season,
                    season=season,
                    league_name=league_name,
                )
            else:
                draft_source = YahooDraftSource(ctx.client, mapper)
                derive_and_store_keeper_costs(
                    draft_source=draft_source,
                    roster_source=roster_source,
                    team_repo=ctx.yahoo_team_repo,
                    keeper_repo=ctx.keeper_repo,
                    league_key=league_key,
                    prior_league_key=prior_league_key,
                    prior_season=prior_season,
                    season=season,
                    league_name=league_name,
                    cost_floor=cost_floor,
                )
            ctx.conn.commit()
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

        # Display results
        all_costs = ctx.keeper_repo.find_by_season_league(season, league_name)
        player_ids = [kc.player_id for kc in all_costs]
        players_list = ctx.player_repo.get_by_ids(player_ids)

        # Build player-to-team mapping from rosters
        teams = ctx.yahoo_team_repo.get_by_league_key(prior_league_key)
        team_name_by_key = {t.team_key: t.name for t in teams}
        player_team: dict[int, str] = {}
        for team in teams:
            try:
                roster = roster_source.fetch_team_roster(
                    team_key=team.team_key,
                    league_key=prior_league_key,
                    season=prior_season,
                    as_of=datetime.date.today(),
                )
                for entry in roster.entries:
                    if entry.player_id is not None:
                        player_team[entry.player_id] = team_name_by_key.get(team.team_key, "?")
            except Exception:
                logger.debug("Failed to fetch roster for team %s", team.team_key)

    if not all_costs:
        console.print("[yellow]No keeper costs derived.[/yellow]")
        return

    player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players_list if p.id is not None}

    table = Table(title=f"Keeper Costs — {league_name} ({season})", show_header=True)
    table.add_column("Team", justify="left")
    table.add_column("Player", justify="left")
    table.add_column("Cost", justify="right")
    table.add_column("Source", justify="left")

    for kc in sorted(all_costs, key=lambda c: (player_team.get(c.player_id, ""), -c.cost)):
        name = player_names.get(kc.player_id, f"ID:{kc.player_id}")
        team_name = player_team.get(kc.player_id, "?")
        table.add_row(team_name, name, f"${kc.cost:.0f}", kc.source)

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
    manager: Annotated[str | None, typer.Option("--manager", help="Team or manager name (default: your team)")] = None,
    estimate_league_keepers: Annotated[
        bool, typer.Option("--estimate-league-keepers", help="Estimate other teams' keepers to adjust surplus")
    ] = False,
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
        prior_season = season - 1
        prior_league_key = _resolve_prior_league_key(ctx, league_key, prior_season)
        prior_game_key = prior_league_key.split(".l.")[0]

        try:
            _ensure_prior_season(ctx, prior_league_key, prior_game_key)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

        mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
        roster_source = YahooRosterSource(ctx.client, mapper)
        try:
            if league_config.keeper_format == "best_n":
                derive_best_n_keeper_costs(
                    roster_source=roster_source,
                    team_repo=ctx.yahoo_team_repo,
                    keeper_repo=ctx.keeper_repo,
                    prior_league_key=prior_league_key,
                    prior_season=prior_season,
                    season=season,
                    league_name=league_name,
                )
            else:
                draft_source = YahooDraftSource(ctx.client, mapper)
                derive_and_store_keeper_costs(
                    draft_source=draft_source,
                    roster_source=roster_source,
                    team_repo=ctx.yahoo_team_repo,
                    keeper_repo=ctx.keeper_repo,
                    league_key=league_key,
                    prior_league_key=prior_league_key,
                    prior_season=prior_season,
                    season=season,
                    league_name=league_name,
                    cost_floor=cost_floor,
                )
            ctx.conn.commit()
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league_name)
        if not keeper_costs:
            console.print("[yellow]No keeper costs found.[/yellow]")
            return

        # Filter to target team's roster
        teams = ctx.yahoo_team_repo.get_by_league_key(prior_league_key)
        if manager is not None:
            manager_lower = manager.lower()
            target_team = next(
                (t for t in teams if manager_lower in t.name.lower() or manager_lower in t.manager_name.lower()),
                None,
            )
            if target_team is None:
                print_error(f"No team matching '{manager}' found.")
                raise typer.Exit(code=1) from None
        else:
            target_team = next((t for t in teams if t.is_owned_by_user), None)
            if target_team is None:
                print_error("No user team found. Use --manager to specify a team.")
                raise typer.Exit(code=1) from None

        target_roster = roster_source.fetch_team_roster(
            team_key=target_team.team_key,
            league_key=prior_league_key,
            season=prior_season,
            as_of=datetime.date.today(),
        )
        roster_player_ids = {e.player_id for e in target_roster.entries if e.player_id is not None}
        keeper_costs = [kc for kc in keeper_costs if kc.player_id in roster_player_ids]

        if not keeper_costs:
            console.print(f"[yellow]No keeper costs for {target_team.name}.[/yellow]")
            return

        console.print(f"[dim]Showing keeper decisions for: {target_team.name}[/dim]\n")

        valuations = ctx.valuation_repo.get_by_season(season, system)
        players = ctx.player_repo.all()

        # Estimate other teams' keepers and adjust valuations
        if estimate_league_keepers:
            max_keepers = league_config.max_keepers
            if max_keepers is None:
                console.print(
                    "[yellow]Warning: max_keepers not configured — skipping league keeper estimation.[/yellow]"
                )
            else:
                other_rosters = fetch_league_rosters(
                    roster_source=roster_source,
                    team_repo=ctx.yahoo_team_repo,
                    prior_league_key=prior_league_key,
                    prior_season=prior_season,
                )
                if other_rosters:
                    fbm_league = load_league(league_name, Path(config_dir))
                    proj_system = valuations[0].projection_system if valuations else "composite"
                    projections = ctx.projection_repo.get_by_season(season, proj_system)

                    eligibility = PlayerEligibilityService(
                        SqlitePositionAppearanceRepo(SingleConnectionProvider(ctx.conn)),
                        pitching_stats_repo=SqlitePitchingStatsRepo(SingleConnectionProvider(ctx.conn)),
                    )
                    batter_positions = eligibility.get_batter_positions(season, fbm_league)
                    pitcher_ids = [p.player_id for p in projections if p.player_type == "pitcher"]
                    pitcher_positions = eligibility.get_pitcher_positions(season, fbm_league, pitcher_ids)

                    original_valuations = valuations
                    valuations = adjust_valuations_for_league_keepers(
                        rosters=other_rosters,
                        valuations=valuations,
                        projections=projections,
                        batter_positions=batter_positions,
                        pitcher_positions=pitcher_positions,
                        league=fbm_league,
                        players=players,
                        max_keepers=max_keepers,
                    )

                    if len(valuations) < len(original_valuations):
                        num_estimated = len(original_valuations) - len(valuations)
                        console.print(
                            f"[dim]Estimated {num_estimated} keepers from {len(other_rosters)} other teams"
                            " — surplus adjusted for draft pool depletion[/dim]"
                        )

    max_keepers = league_config.max_keepers
    decisions = compute_surplus(keeper_costs, valuations, players, threshold=threshold, decay=decay)
    if max_keepers is not None:
        decisions = decisions[:max_keepers]
    print_keeper_decisions(decisions)


@yahoo_app.command("keeper-league")
def yahoo_keeper_league(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    max_keepers_opt: Annotated[int | None, typer.Option("--max-keepers", help="Override max keepers")] = None,
    top_targets: Annotated[int, typer.Option("--top-targets", help="Number of trade targets to show")] = 15,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Show league-wide keeper projections, category comparison, and trade targets."""
    league_name, config = _resolve_league_context(league, config_dir)
    league_config = config.leagues[league_name]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        game_key = ctx.client.get_game_key(season)
        league_key = f"{game_key}.l.{league_config.league_id}"

        prior_season = season - 1
        prior_league_key = _resolve_prior_league_key(ctx, league_key, prior_season)
        prior_game_key = prior_league_key.split(".l.")[0]

        try:
            _ensure_prior_season(ctx, prior_league_key, prior_game_key)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

        # Determine max_keepers
        max_keepers = max_keepers_opt or league_config.max_keepers
        if max_keepers is None:
            print_error("max_keepers not configured and --max-keepers not provided")
            raise typer.Exit(code=1)

        # Get rosters (DB-first, auto-sync if missing)
        rosters = ctx.yahoo_roster_repo.get_by_league_latest(prior_league_key)
        if not rosters:
            mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
            roster_source = YahooRosterSource(ctx.client, mapper)
            teams = ctx.yahoo_team_repo.get_by_league_key(prior_league_key)
            today = datetime.date.today()
            for team in teams:
                roster = roster_source.fetch_team_roster(
                    team_key=team.team_key,
                    league_key=prior_league_key,
                    season=prior_season,
                    as_of=today,
                )
                ctx.yahoo_roster_repo.save_snapshot(roster)
            ctx.conn.commit()
            rosters = ctx.yahoo_roster_repo.get_by_league_latest(prior_league_key)

        # Build team_names and find user_team_key
        teams = ctx.yahoo_team_repo.get_by_league_key(prior_league_key)
        team_names = {t.team_key: t.name for t in teams}
        user_team = next((t for t in teams if t.is_owned_by_user), None)
        if user_team is None:
            print_error("Could not identify user's team in the league")
            raise typer.Exit(code=1)
        user_team_key = user_team.team_key

        # Get valuations and players
        valuations = ctx.valuation_repo.get_by_season(season, system)
        if not valuations:
            print_error(f"No valuations found for season {season}, system '{system}'")
            raise typer.Exit(code=1)
        players = ctx.player_repo.all()

        overview = build_league_keeper_overview(
            rosters=rosters,
            valuations=valuations,
            players=players,
            max_keepers=max_keepers,
            user_team_key=user_team_key,
            team_names=team_names,
        )
        print_league_keeper_overview(overview, top_targets=top_targets)


@yahoo_app.command("draft-needs")
def yahoo_draft_needs(  # pragma: no cover
    league: Annotated[str | None, typer.Option("--league", help="League name from [yahoo.leagues]")] = None,
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    max_keepers_opt: Annotated[int | None, typer.Option("--max-keepers", help="Override max keepers")] = None,
    top: Annotated[int, typer.Option("--top", help="Number of recommendations per category")] = 5,
    data_dir: _DataDirOpt = "./data",
    config_dir: Annotated[str, typer.Option("--config-dir", help="Config directory")] = ".",
) -> None:
    """Show category gaps and draft recommendations based on projected keepers."""
    league_name, config = _resolve_league_context(league, config_dir)
    league_config = config.leagues[league_name]

    with build_yahoo_context(data_dir, Path(config_dir)) as ctx:
        game_key = ctx.client.get_game_key(season)
        league_key = f"{game_key}.l.{league_config.league_id}"

        prior_season = season - 1
        prior_league_key = _resolve_prior_league_key(ctx, league_key, prior_season)
        prior_game_key = prior_league_key.split(".l.")[0]

        try:
            _ensure_prior_season(ctx, prior_league_key, prior_game_key)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None

        # Determine max_keepers
        max_keepers = max_keepers_opt or league_config.max_keepers
        if max_keepers is None:
            print_error("max_keepers not configured and --max-keepers not provided")
            raise typer.Exit(code=1)

        # Get rosters (DB-first, auto-sync if missing)
        rosters = ctx.yahoo_roster_repo.get_by_league_latest(prior_league_key)
        if not rosters:
            mapper = YahooPlayerMapper(ctx.yahoo_player_map_repo, ctx.player_repo)
            roster_source = YahooRosterSource(ctx.client, mapper)
            teams = ctx.yahoo_team_repo.get_by_league_key(prior_league_key)
            today = datetime.date.today()
            for team in teams:
                roster = roster_source.fetch_team_roster(
                    team_key=team.team_key,
                    league_key=prior_league_key,
                    season=prior_season,
                    as_of=today,
                )
                ctx.yahoo_roster_repo.save_snapshot(roster)
            ctx.conn.commit()
            rosters = ctx.yahoo_roster_repo.get_by_league_latest(prior_league_key)

        # Build team_names and find user_team_key
        teams = ctx.yahoo_team_repo.get_by_league_key(prior_league_key)
        team_names = {t.team_key: t.name for t in teams}
        user_team = next((t for t in teams if t.is_owned_by_user), None)
        if user_team is None:
            print_error("Could not identify user's team in the league")
            raise typer.Exit(code=1)
        user_team_key = user_team.team_key

        # Get valuations, players, projections
        valuations = ctx.valuation_repo.get_by_season(season, system)
        if not valuations:
            print_error(f"No valuations found for season {season}, system '{system}'")
            raise typer.Exit(code=1)
        players = ctx.player_repo.all()

        proj_system = valuations[0].projection_system if valuations else "composite"
        projections = ctx.projection_repo.get_by_season(season, proj_system)

        fbm_league = load_league(league_name, Path(config_dir))

        # Build overview for display
        overview = build_league_keeper_overview(
            rosters=rosters,
            valuations=valuations,
            players=players,
            max_keepers=max_keepers,
            user_team_key=user_team_key,
            team_names=team_names,
        )

        # Build needs analysis
        analysis, needs = build_keeper_draft_needs(
            rosters=rosters,
            valuations=valuations,
            players=players,
            projections=projections,
            max_keepers=max_keepers,
            user_team_key=user_team_key,
            team_names=team_names,
            league=fbm_league,
            top_n=top,
        )

        print_keeper_draft_needs(overview, analysis, needs, fbm_league.teams)


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
        matches = resolve_players(ctx.player_repo, player)
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
