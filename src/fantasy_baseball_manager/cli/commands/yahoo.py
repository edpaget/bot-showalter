import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from fantasy_baseball_manager.cli._output import console, print_error
from fantasy_baseball_manager.cli.factory import build_yahoo_context
from fantasy_baseball_manager.config_yahoo import YahooConfigError, load_yahoo_config, resolve_default_league
from fantasy_baseball_manager.domain.yahoo_league import YahooLeague, YahooTeam
from fantasy_baseball_manager.domain.yahoo_player import YahooPlayerMap
from fantasy_baseball_manager.yahoo.auth import YahooAuth
from fantasy_baseball_manager.yahoo.league_source import YahooLeagueSource
from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper
from fantasy_baseball_manager.yahoo.roster_source import YahooRosterSource

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
