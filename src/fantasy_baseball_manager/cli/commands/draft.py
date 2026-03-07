import csv
import functools
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

from fantasy_baseball_manager.cli._live_server import create_live_draft_app
from fantasy_baseball_manager.cli._output import (
    console,
    print_batch_simulation_result,
    print_cascade_result,
    print_category_needs,
    print_draft_board,
    print_draft_report,
    print_draft_tiers,
    print_pick_trade_evaluation,
    print_pick_value_curve,
    print_position_check,
    print_scarcity_rankings,
    print_scarcity_report,
    print_tier_summary,
    print_upgrades,
    print_value_curve,
)
from fantasy_baseball_manager.cli.factory import (
    DraftBoardContext,
    build_category_needs_context,
    build_draft_board_context,
)
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.config_yahoo import YahooConfigError, load_yahoo_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain import DraftBoard, DraftBoardRow, PickTrade, Valuation
from fantasy_baseball_manager.name_utils import resolve_players
from fantasy_baseball_manager.repos import SqliteDraftSessionRepo
from fantasy_baseball_manager.services import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftSession,
    PlayerEligibilityService,
    adjust_valuations_for_league_keepers,
    build_draft_board,
    build_draft_roster_slots,
    build_roster_state,
    cascade_analysis,
    compute_category_balance_scores,
    compute_marginal_values,
    compute_opportunity_costs,
    compute_pick_value_curve,
    compute_position_upgrades,
    compute_scarcity,
    compute_scarcity_rankings,
    compute_value_curves,
    evaluate_pick_trade,
    export_csv,
    export_html,
    generate_tiers,
    identify_needs,
    load_draft,
    optimize_auction_budget,
    parse_league_keepers,
    plan_snake_draft,
    recommend,
    simulate_drafts,
    tier_summary,
)
from fantasy_baseball_manager.services import (
    draft_report as compute_draft_report,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import ADP, LeagueSettings
    from fantasy_baseball_manager.repos import SqlitePlayerRepo

draft_app = typer.Typer(name="draft", help="Draft board display and export")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


def _apply_keeper_exclusion(  # pragma: no cover
    ctx: DraftBoardContext,
    valuations: list[Valuation],
    exclude_keepers: str,
    season: int,
) -> list[Valuation]:
    """Remove estimated league keepers from valuations and recalculate values."""
    try:
        yahoo_config = load_yahoo_config(Path.cwd())
    except YahooConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if exclude_keepers not in yahoo_config.leagues:
        console.print(f"[red]Yahoo league '{exclude_keepers}' not found in [yahoo.leagues][/red]")
        raise typer.Exit(code=1)

    league_config = yahoo_config.leagues[exclude_keepers]
    max_keepers = league_config.max_keepers
    if max_keepers is None:
        console.print(f"[red]max_keepers not configured for Yahoo league '{exclude_keepers}'[/red]")
        raise typer.Exit(code=1)

    # Find the current-season league key by matching league_id
    all_leagues = ctx.yahoo_league_repo.get_all()
    current_league = next(
        (lg for lg in all_leagues if lg.season == season and str(league_config.league_id) in lg.league_key),
        None,
    )
    if current_league is None:
        console.print(
            f"[red]No Yahoo league found for season {season}. Run `yahoo sync --league {exclude_keepers}` first.[/red]"
        )
        raise typer.Exit(code=1)

    if current_league.renew is None:
        console.print(f"[red]No renew chain for league {current_league.league_key} — cannot find prior season.[/red]")
        raise typer.Exit(code=1)

    prior_league_key = current_league.renew.replace("_", ".l.", 1)

    rosters = ctx.yahoo_roster_repo.get_by_league_latest(prior_league_key)
    if not rosters:
        console.print(
            f"[red]No rosters found for prior league {prior_league_key}."
            " Run `yahoo rosters` or `yahoo keeper-league` first.[/red]"
        )
        raise typer.Exit(code=1)

    fbm_league = load_league(exclude_keepers, Path.cwd())
    proj_system = valuations[0].projection_system if valuations else "composite"
    projections = ctx.projection_repo.get_by_season(season, proj_system)

    eligibility = PlayerEligibilityService(
        ctx.position_appearance_repo,
        pitching_stats_repo=ctx.pitching_stats_repo,
    )
    batter_positions = eligibility.get_batter_positions(season, fbm_league)
    pitcher_ids = [p.player_id for p in projections if p.player_type == "pitcher"]
    pitcher_positions = eligibility.get_pitcher_positions(season, fbm_league, pitcher_ids)

    players = ctx.player_repo.all()

    original_count = len(valuations)
    valuations = adjust_valuations_for_league_keepers(
        rosters=rosters,
        valuations=valuations,
        projections=projections,
        batter_positions=batter_positions,
        pitcher_positions=pitcher_positions,
        league=fbm_league,
        players=players,
        max_keepers=max_keepers,
    )

    excluded = original_count - len(valuations)
    if excluded > 0:
        console.print(
            f"[dim]Excluded {excluded} estimated keepers from {len(rosters)} teams"
            " — valuations adjusted for draft pool depletion[/dim]"
        )

    return valuations


def _fetch_draft_board_data(
    season: int,
    system: str,
    version: str,
    league_name: str,
    provider: str,
    data_dir: str,
    player_type: str | None,
    position: str | None,
    top: int | None,
    exclude_keepers: str | None = None,
) -> tuple[DraftBoard, LeagueSettings]:
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        if exclude_keepers is not None:
            valuations = _apply_keeper_exclusion(ctx, valuations, exclude_keepers, season)

        if player_type is not None:
            valuations = [v for v in valuations if v.player_type == player_type]
        if position is not None:
            valuations = [v for v in valuations if v.position == position]

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)
        profiles = ctx.profile_service.enrich_valuations(valuations, season)

        board = build_draft_board(
            valuations, league, player_names, adp=adp_list if adp_list else None, profiles=profiles
        )

        if top is not None:
            board = DraftBoard(
                rows=board.rows[:top],
                batting_categories=board.batting_categories,
                pitching_categories=board.pitching_categories,
            )

    return board, league


@draft_app.command("board")
def draft_board(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    player_type: Annotated[str | None, typer.Option("--player-type", help="Filter by player type")] = None,
    position: Annotated[str | None, typer.Option("--position", help="Filter by position")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    exclude_keepers: Annotated[
        str | None, typer.Option("--exclude-keepers", help="Yahoo league name to exclude keepers from")
    ] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display the draft board in the terminal."""
    board, _league = _fetch_draft_board_data(
        season, system, version, league_name, provider, data_dir, player_type, position, top, exclude_keepers
    )
    print_draft_board(board)


@draft_app.command("export")
def draft_export(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    output: Annotated[Path, typer.Option("--output", help="Output file path")],
    fmt: Annotated[str, typer.Option("--format", help="Output format: csv or html")] = "csv",
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    player_type: Annotated[str | None, typer.Option("--player-type", help="Filter by player type")] = None,
    position: Annotated[str | None, typer.Option("--position", help="Filter by position")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    exclude_keepers: Annotated[
        str | None, typer.Option("--exclude-keepers", help="Yahoo league name to exclude keepers from")
    ] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Export the draft board to a file (CSV or HTML)."""
    board, league = _fetch_draft_board_data(
        season, system, version, league_name, provider, data_dir, player_type, position, top, exclude_keepers
    )
    with open(output, "w", newline="", encoding="utf-8") as f:
        if fmt == "html":
            export_html(board, league, f)
        else:
            export_csv(board, f)
    console.print(f"Draft board exported to {output}")


@draft_app.command("live")
def draft_live(  # pragma: no cover
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    player_type: Annotated[str | None, typer.Option("--player-type", help="Filter by player type")] = None,
    position: Annotated[str | None, typer.Option("--position", help="Filter by position")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    host: Annotated[str, typer.Option("--host", help="Server host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Server port")] = 5000,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Start a live draft server with auto-refreshing HTML board."""
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]
        if player_type is not None:
            valuations = [v for v in valuations if v.player_type == player_type]
        if position is not None:
            valuations = [v for v in valuations if v.position == position]

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)
        profiles = ctx.profile_service.enrich_valuations(valuations, season)

    if top is not None:
        valuations = sorted(valuations, key=lambda v: v.value, reverse=True)[:top]

    flask_app = create_live_draft_app(
        valuations, league, player_names, adp=adp_list if adp_list else None, profiles=profiles
    )
    console.print(f"Live draft server running at http://{host}:{port}/")
    flask_app.run(host=host, port=port)


@draft_app.command("start")
def draft_start(  # pragma: no cover
    season: Annotated[int, typer.Option("--season", help="Season year")],
    teams: Annotated[int, typer.Option("--teams", help="Number of teams in league")],
    slot: Annotated[int, typer.Option("--slot", help="Your draft slot (1-based)")],
    fmt: Annotated[str, typer.Option("--format", help="Draft format: snake or auction")] = "snake",
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    budget: Annotated[int, typer.Option("--budget", help="Auction budget per team")] = 260,
    resume: Annotated[Path | None, typer.Option("--resume", help="Resume from saved draft file")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Start an interactive draft session (REPL)."""
    league = load_league(league_name, Path.cwd())

    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        player_ids = [v.player_id for v in valuations]
        players_list = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players_list if p.id is not None}

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)
        profiles = ctx.profile_service.enrich_valuations(valuations, season)

        projections = ctx.projection_repo.get_by_season(season, "steamer")

    board = build_draft_board(valuations, league, player_names, adp=adp_list if adp_list else None, profiles=profiles)
    draft_players: list[DraftBoardRow] = board.rows
    roster_slots = build_draft_roster_slots(league)

    draft_format = DraftFormat(fmt)
    config = DraftConfig(
        teams=teams,
        roster_slots=roster_slots,
        format=draft_format,
        user_team=slot,
        season=season,
        budget=budget if draft_format == DraftFormat.AUCTION else 0,
    )

    if resume is not None:
        engine = load_draft(resume, draft_players)
    else:
        engine = DraftEngine()
        engine.start(draft_players, config)

    save_path = resume or Path(f"draft-{league_name}-{season}.json")

    report_fn = functools.partial(
        compute_draft_report,
        batting_categories=board.batting_categories,
        pitching_categories=board.pitching_categories,
    )

    def _cat_balance_fn(roster_ids: list[int], available_ids: list[int]) -> dict[int, float]:
        return compute_category_balance_scores(roster_ids, available_ids, projections, league)

    recommend_fn = functools.partial(recommend, category_balance_fn=_cat_balance_fn)

    db_conn = create_connection(Path(data_dir) / "fbm.db")
    session_repo = SqliteDraftSessionRepo(SingleConnectionProvider(db_conn))

    session = DraftSession(
        engine=engine,
        players=draft_players,
        console=console,
        recommend_fn=recommend_fn,
        report_fn=report_fn,
        save_path=save_path,
        projections=projections,
        league=league,
        session_repo=session_repo,
        league_name=league_name,
    )
    session.run()


@draft_app.command("report")
def draft_report_cmd(
    draft_file: Annotated[Path, typer.Argument(help="Path to saved draft JSON file")],
    season: Annotated[int, typer.Option("--season", help="Season year")] = 2026,
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Generate a post-draft analysis report from a saved draft file."""
    league = load_league(league_name, Path.cwd())

    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        player_ids = [v.player_id for v in valuations]
        players_list = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players_list if p.id is not None}

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)
        profiles = ctx.profile_service.enrich_valuations(valuations, season)

    board = build_draft_board(valuations, league, player_names, adp=adp_list if adp_list else None, profiles=profiles)
    draft_players: list[DraftBoardRow] = board.rows

    engine = load_draft(draft_file, draft_players)
    report = compute_draft_report(
        engine.state,
        draft_players,
        batting_categories=board.batting_categories,
        pitching_categories=board.pitching_categories,
    )
    print_draft_report(report)


@draft_app.command("tiers")
def draft_tiers(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    method: Annotated[str, typer.Option("--method", help="Tiering method: gap or jenks")] = "gap",
    max_tiers: Annotated[int, typer.Option("--max-tiers", help="Max tiers per position")] = 5,
    position: Annotated[str | None, typer.Option("--position", help="Filter to a single position")] = None,
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    exclude_keepers: Annotated[
        str | None, typer.Option("--exclude-keepers", help="Yahoo league name to exclude keepers from")
    ] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display position-grouped tier assignments."""
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        if exclude_keepers is not None:
            valuations = _apply_keeper_exclusion(ctx, valuations, exclude_keepers, season)

        if position is not None:
            valuations = [v for v in valuations if v.position == position]

        tiers = generate_tiers(valuations, ctx.player_repo, method=method, max_tiers=max_tiers)

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)
        adp_by_player: dict[int, ADP] | None = None
        if adp_list:
            adp_by_player = {}
            for adp in adp_list:
                adp_by_player[adp.player_id] = adp

        print_draft_tiers(tiers, adp_by_player=adp_by_player)


@draft_app.command("tier-summary")
def draft_tier_summary(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    method: Annotated[str, typer.Option("--method", help="Tiering method: gap or jenks")] = "gap",
    max_tiers: Annotated[int, typer.Option("--max-tiers", help="Max tiers per position")] = 5,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display a cross-position tier summary matrix."""
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        tiers = generate_tiers(valuations, ctx.player_repo, method=method, max_tiers=max_tiers)
        report = tier_summary(tiers)
        print_tier_summary(report)


@draft_app.command("budget")
def budget_command(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    strategy: Annotated[str, typer.Option("--strategy", help="balanced or stars_and_scrubs")] = "balanced",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    method: Annotated[str, typer.Option("--method", help="Tiering method: gap or jenks")] = "gap",
    max_tiers: Annotated[int, typer.Option("--max-tiers", help="Max tiers per position")] = 5,
    data_dir: _DataDirOpt = "./data",
) -> None:  # pragma: no cover
    """Compute optimal auction budget allocation across roster positions."""
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        tiers = generate_tiers(valuations, ctx.player_repo, method=method, max_tiers=max_tiers)

        allocations = optimize_auction_budget(valuations, league, player_names, strategy=strategy, tiers=tiers)

    table = Table(title=f"Auction Budget — {strategy}")
    table.add_column("Position", style="bold")
    table.add_column("Budget", justify="right")
    table.add_column("Tier", justify="center")
    table.add_column("Target Players")

    for alloc in sorted(allocations, key=lambda a: a.budget, reverse=True):
        tier_str = str(alloc.target_tier) if alloc.target_tier is not None else "-"
        names_str = ", ".join(alloc.target_player_names) if alloc.target_player_names else "-"
        table.add_row(alloc.position.upper(), f"${alloc.budget:.0f}", tier_str, names_str)

    table.add_section()
    table.add_row("TOTAL", f"${sum(a.budget for a in allocations):.0f}", "", "")
    console.print(table)


@draft_app.command("plan")
def plan_command(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    slot: Annotated[int, typer.Option("--slot", help="Your draft slot (1-indexed)")],
    teams: Annotated[int, typer.Option("--teams", help="Number of teams")] = 12,
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    method: Annotated[str, typer.Option("--method", help="Tiering method: gap or jenks")] = "gap",
    max_tiers: Annotated[int, typer.Option("--max-tiers", help="Max tiers per position")] = 5,
    keepers: Annotated[str | None, typer.Option("--keepers", help="Your keepers: 'Name:pos,Name:pos'")] = None,
    league_keepers: Annotated[Path | None, typer.Option("--league-keepers", help="CSV of all league keepers")] = None,
    keepers_per_team: Annotated[int, typer.Option("--keepers-per-team", help="Keepers per team")] = 0,
    data_dir: _DataDirOpt = "./data",
) -> None:  # pragma: no cover
    """Compute a round-by-round snake draft plan for a given draft slot."""
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}
        name_to_id = {name: pid for pid, name in player_names.items()}

        tiers = generate_tiers(valuations, ctx.player_repo, method=method, max_tiers=max_tiers)

        my_keepers_list: list[tuple[int, str]] | None = None
        if keepers is not None:
            my_keepers_list = []
            for entry in keepers.split(","):
                name, pos = entry.rsplit(":", 1)
                pid = name_to_id.get(name.strip())
                if pid is None:
                    console.print(f"[yellow]Keeper '{name.strip()}' not found, skipping[/yellow]")
                    continue
                my_keepers_list.append((pid, pos.strip().lower()))

        league_keeper_ids: set[int] | None = None
        if league_keepers is not None:
            with open(league_keepers, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            all_players = ctx.player_repo.all()
            league_keeper_ids, _unmatched = parse_league_keepers(rows, all_players)

        plan = plan_snake_draft(
            valuations,
            league,
            player_names,
            draft_slot=slot,
            tiers=tiers,
            my_keepers=my_keepers_list,
            league_keeper_ids=league_keeper_ids,
            keepers_per_team=keepers_per_team,
        )

    table = Table(title=f"Snake Draft Plan — Slot {slot}/{teams}")
    table.add_column("Round", justify="right")
    table.add_column("Pick#", justify="right")
    table.add_column("Position", style="bold")
    table.add_column("Tier", justify="center")
    table.add_column("Value", justify="right")
    table.add_column("Alternatives")

    for rnd in plan.rounds:
        tier_str = str(rnd.target_tier) if rnd.target_tier is not None else "-"
        alt_str = ", ".join(p.upper() for p in rnd.alternative_positions) if rnd.alternative_positions else "-"
        table.add_row(
            str(rnd.round),
            str(rnd.pick_number),
            rnd.recommended_position.upper(),
            tier_str,
            f"{rnd.expected_value:.1f}",
            alt_str,
        )

    console.print(table)


@draft_app.command("simulate")
def simulate_command(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    slot: Annotated[int, typer.Option("--slot", help="Your draft slot (1-indexed)")],
    simulations: Annotated[int, typer.Option("--simulations", help="Number of simulations")] = 1000,
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    keepers: Annotated[str | None, typer.Option("--keepers", help="Your keepers: 'Name:pos,Name:pos'")] = None,
    league_keepers: Annotated[Path | None, typer.Option("--league-keepers", help="CSV of all league keepers")] = None,
    keepers_per_team: Annotated[int, typer.Option("--keepers-per-team", help="Keepers per team")] = 0,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:  # pragma: no cover
    """Run Monte Carlo draft simulations to estimate expected roster value."""
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}
        name_to_id = {name: pid for pid, name in player_names.items()}

        my_keepers_list: list[tuple[int, str]] | None = None
        if keepers is not None:
            my_keepers_list = []
            for entry in keepers.split(","):
                name, pos = entry.rsplit(":", 1)
                pid = name_to_id.get(name.strip())
                if pid is None:
                    console.print(f"[yellow]Keeper '{name.strip()}' not found, skipping[/yellow]")
                    continue
                my_keepers_list.append((pid, pos.strip().upper()))

        league_keeper_ids: set[int] | None = None
        if league_keepers is not None:
            with open(league_keepers, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            all_players = ctx.player_repo.all()
            league_keeper_ids, _unmatched = parse_league_keepers(rows, all_players)

        result = simulate_drafts(
            valuations,
            league,
            player_names,
            draft_slot=slot,
            n_simulations=simulations,
            my_keepers=my_keepers_list,
            league_keeper_ids=league_keeper_ids,
            keepers_per_team=keepers_per_team,
            seed=seed,
        )

    print_batch_simulation_result(result)


@draft_app.command("needs")
def draft_needs(
    roster: Annotated[str, typer.Option("--roster", help="Comma-separated player names")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Projection system")] = "steamer",
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Identify category weaknesses and recommend available players."""
    league = load_league(league_name, Path.cwd())
    roster_names = [name.strip() for name in roster.split(",")]

    with build_category_needs_context(data_dir) as ctx:
        roster_ids = _resolve_roster_names(roster_names, ctx.player_repo)

        projections = ctx.projection_repo.get_by_season(season, system)
        all_projected_ids = {p.player_id for p in projections}
        available_ids = [pid for pid in all_projected_ids if pid not in set(roster_ids)]

        players = ctx.player_repo.get_by_ids(roster_ids + available_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        needs = identify_needs(roster_ids, available_ids, projections, league, player_names=player_names)
        print_category_needs(needs, league.teams)


@draft_app.command("pick-values")
def draft_pick_values(
    season: Annotated[int, typer.Option("--season")],
    system: Annotated[str, typer.Option("--system")] = "zar",
    version: Annotated[str, typer.Option("--version")] = "1.0",
    provider: Annotated[str, typer.Option("--provider")] = "fantasypros",
    league_name: Annotated[str, typer.Option("--league")] = "default",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display pick value curve mapping draft picks to expected player value."""
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)

        curve = compute_pick_value_curve(adp_list, valuations, league, player_names=player_names)
    print_pick_value_curve(curve)


@draft_app.command("trade-picks")
def draft_trade_picks(
    gives: Annotated[str, typer.Option("--gives", help="Comma-separated pick numbers")],
    receives: Annotated[str, typer.Option("--receives", help="Comma-separated pick numbers")],
    season: Annotated[int, typer.Option("--season")],
    system: Annotated[str, typer.Option("--system")] = "zar",
    version: Annotated[str, typer.Option("--version")] = "1.0",
    provider: Annotated[str, typer.Option("--provider")] = "fantasypros",
    league_name: Annotated[str, typer.Option("--league")] = "default",
    cascade: Annotated[bool, typer.Option("--cascade")] = False,
    team: Annotated[int, typer.Option("--team", help="User team index (0-based)")] = 0,
    threshold: Annotated[float, typer.Option("--threshold")] = 1.0,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate a draft pick trade by comparing expected value on each side."""
    give_picks = [int(p.strip()) for p in gives.split(",")]
    receive_picks = [int(p.strip()) for p in receives.split(",")]
    trade = PickTrade(gives=give_picks, receives=receive_picks)

    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)

        curve = compute_pick_value_curve(adp_list, valuations, league, player_names=player_names)
        evaluation = evaluate_pick_trade(trade, curve, threshold=threshold)
        print_pick_trade_evaluation(evaluation)

        if cascade:
            profiles = ctx.profile_service.enrich_valuations(valuations, season)
            board = build_draft_board(
                valuations, league, player_names, adp=adp_list if adp_list else None, profiles=profiles
            )
            result = cascade_analysis(trade, board, league, team, threshold=threshold)
            print_cascade_result(result)


@draft_app.command("scarcity")
def draft_scarcity(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    position: Annotated[str | None, typer.Option("--position", help="Filter to a single position")] = None,
    detail: Annotated[bool, typer.Option("--detail", help="Show full value curve instead of summary")] = False,
    exclude_keepers: Annotated[
        str | None, typer.Option("--exclude-keepers", help="Yahoo league name to exclude keepers from")
    ] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display positional scarcity analysis ranked by dropoff severity."""
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        if exclude_keepers is not None:
            valuations = _apply_keeper_exclusion(ctx, valuations, exclude_keepers, season)

        if detail:
            player_ids = [v.player_id for v in valuations]
            players = ctx.player_repo.get_by_ids(player_ids)
            player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}
            curves = compute_value_curves(valuations, league, player_names)
            if position is not None:
                curves = [c for c in curves if c.position == position]
            for curve in curves:
                print_value_curve(curve, league)
        else:
            scarcities = compute_scarcity(valuations, league)
            if position is not None:
                scarcities = [s for s in scarcities if s.position == position]
            print_scarcity_report(scarcities, league)


@draft_app.command("scarcity-rankings")
def draft_scarcity_rankings(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    exclude_keepers: Annotated[
        str | None, typer.Option("--exclude-keepers", help="Yahoo league name to exclude keepers from")
    ] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display scarcity-adjusted player rankings."""
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        if exclude_keepers is not None:
            valuations = _apply_keeper_exclusion(ctx, valuations, exclude_keepers, season)

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        rankings = compute_scarcity_rankings(valuations, league, player_names)

    if top is not None:
        rankings = rankings[:top]
    print_scarcity_rankings(rankings, league)


def _resolve_roster_names(roster_names: list[str], player_repo: SqlitePlayerRepo) -> list[int]:
    """Resolve player names to IDs, printing warnings for ambiguous/missing names."""
    roster_ids: list[int] = []
    for name in roster_names:
        matches = resolve_players(player_repo, name)
        if len(matches) == 1 and matches[0].id is not None:
            roster_ids.append(matches[0].id)
        elif len(matches) > 1:
            console.print(f"[yellow]Ambiguous name '{name}', skipping[/yellow]")
        else:
            console.print(f"[yellow]Player '{name}' not found, skipping[/yellow]")
    return roster_ids


def _parse_roster_option(
    roster: str | None,
    roster_file: Path | None,
) -> list[str]:
    """Parse roster names from --roster or --roster-file options."""
    if roster is not None:
        return [name.strip() for name in roster.split(",")]
    if roster_file is not None:
        return [line.strip() for line in roster_file.read_text().splitlines() if line.strip()]
    console.print("[red]Either --roster or --roster-file is required.[/red]")
    raise typer.Exit(code=1)


@draft_app.command("upgrades")
def draft_upgrades(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    roster: Annotated[str | None, typer.Option("--roster", help="Comma-separated player names")] = None,
    roster_file: Annotated[
        Path | None, typer.Option("--roster-file", help="File with one player name per line")
    ] = None,
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    top: Annotated[int, typer.Option("--top", help="Show top N players")] = 10,
    opportunity_cost: Annotated[
        bool, typer.Option("--opportunity-cost", help="Include opportunity cost analysis")
    ] = False,
    picks_until_next: Annotated[int, typer.Option("--picks-until-next", help="Picks between your turns")] = 12,
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    exclude_keepers: Annotated[
        str | None, typer.Option("--exclude-keepers", help="Yahoo league name to exclude keepers from")
    ] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Re-rank available players by marginal value given your current roster."""
    roster_names = _parse_roster_option(roster, roster_file)
    league = load_league(league_name, Path.cwd())

    with build_draft_board_context(data_dir) as ctx:
        roster_ids = _resolve_roster_names(roster_names, ctx.player_repo)

        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        if exclude_keepers is not None:
            valuations = _apply_keeper_exclusion(ctx, valuations, exclude_keepers, season)

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)
        profiles = ctx.profile_service.enrich_valuations(valuations, season)

    board = build_draft_board(valuations, league, player_names, adp=adp_list if adp_list else None, profiles=profiles)

    state = build_roster_state(roster_ids, board, league)
    roster_id_set = set(roster_ids)
    available = [r for r in board.rows if r.player_id not in roster_id_set]
    marginal_values = compute_marginal_values(state, available, league)

    opp_costs = None
    if opportunity_cost:
        opp_costs = compute_opportunity_costs(marginal_values, state, league, picks_until_next)

    print_upgrades(marginal_values[:top], opportunity_costs=opp_costs)


@draft_app.command("position-check")
def draft_position_check(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    roster: Annotated[str | None, typer.Option("--roster", help="Comma-separated player names")] = None,
    roster_file: Annotated[
        Path | None, typer.Option("--roster-file", help="File with one player name per line")
    ] = None,
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show per-position upgrade comparison for your current roster."""
    roster_names = _parse_roster_option(roster, roster_file)
    league = load_league(league_name, Path.cwd())

    with build_draft_board_context(data_dir) as ctx:
        roster_ids = _resolve_roster_names(roster_names, ctx.player_repo)

        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        adp_list = ctx.adp_repo.get_by_season(season, provider=provider)
        profiles = ctx.profile_service.enrich_valuations(valuations, season)

    board = build_draft_board(valuations, league, player_names, adp=adp_list if adp_list else None, profiles=profiles)

    state = build_roster_state(roster_ids, board, league)
    roster_id_set = set(roster_ids)
    available = [r for r in board.rows if r.player_id not in roster_id_set]
    upgrades = compute_position_upgrades(state, available, league)
    print_position_check(upgrades)
