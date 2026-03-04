import functools
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from fantasy_baseball_manager.cli._live_server import create_live_draft_app
from fantasy_baseball_manager.cli._output import (
    console,
    print_cascade_result,
    print_category_needs,
    print_draft_board,
    print_draft_report,
    print_draft_tiers,
    print_pick_trade_evaluation,
    print_pick_value_curve,
    print_scarcity_rankings,
    print_scarcity_report,
    print_tier_summary,
    print_value_curve,
)
from fantasy_baseball_manager.cli.factory import build_category_needs_context, build_draft_board_context
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain import DraftBoard, DraftBoardRow, PickTrade
from fantasy_baseball_manager.services import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftSession,
    build_draft_board,
    build_draft_roster_slots,
    cascade_analysis,
    compute_category_balance_scores,
    compute_pick_value_curve,
    compute_scarcity,
    compute_scarcity_rankings,
    compute_value_curves,
    evaluate_pick_trade,
    export_csv,
    export_html,
    generate_tiers,
    identify_needs,
    load_draft,
    recommend,
    tier_summary,
)
from fantasy_baseball_manager.services import (
    draft_report as compute_draft_report,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import ADP, LeagueSettings
draft_app = typer.Typer(name="draft", help="Draft board display and export")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


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
) -> tuple[DraftBoard, LeagueSettings]:
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
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display the draft board in the terminal."""
    board, _league = _fetch_draft_board_data(
        season, system, version, league_name, provider, data_dir, player_type, position, top
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
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Export the draft board to a file (CSV or HTML)."""
    board, league = _fetch_draft_board_data(
        season, system, version, league_name, provider, data_dir, player_type, position, top
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

    session = DraftSession(
        engine=engine,
        players=draft_players,
        console=console,
        recommend_fn=recommend_fn,
        report_fn=report_fn,
        save_path=save_path,
        projections=projections,
        league=league,
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
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display position-grouped tier assignments."""
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]
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
        roster_ids: list[int] = []
        for name in roster_names:
            matches = ctx.player_repo.search_by_name(name)
            if len(matches) == 1 and matches[0].id is not None:
                roster_ids.append(matches[0].id)
            elif len(matches) > 1:
                console.print(f"[yellow]Ambiguous name '{name}', skipping[/yellow]")
            else:
                console.print(f"[yellow]Player '{name}' not found, skipping[/yellow]")

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
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display positional scarcity analysis ranked by dropoff severity."""
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

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
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Display scarcity-adjusted player rankings."""
    league = load_league(league_name, Path.cwd())
    with build_draft_board_context(data_dir) as ctx:
        valuations = ctx.valuation_repo.get_by_season(season, system=system)
        valuations = [v for v in valuations if v.version == version]

        player_ids = [v.player_id for v in valuations]
        players = ctx.player_repo.get_by_ids(player_ids)
        player_names = {p.id: f"{p.name_first} {p.name_last}" for p in players if p.id is not None}

        rankings = compute_scarcity_rankings(valuations, league, player_names)

    if top is not None:
        rankings = rankings[:top]
    print_scarcity_rankings(rankings, league)
