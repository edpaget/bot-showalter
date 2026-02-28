import functools
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from fantasy_baseball_manager.cli._live_server import create_live_draft_app
from fantasy_baseball_manager.cli._output import (
    console,
    print_draft_board,
    print_draft_report,
    print_draft_tiers,
    print_tier_summary,
)
from fantasy_baseball_manager.cli.factory import build_draft_board_context
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain import DraftBoard, DraftBoardRow
from fantasy_baseball_manager.services import (
    DraftConfig,
    DraftEngine,
    DraftFormat,
    DraftSession,
    build_draft_board,
    build_draft_roster_slots,
    export_csv,
    export_html,
    generate_tiers,
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

    session = DraftSession(
        engine=engine,
        players=draft_players,
        console=console,
        recommend_fn=recommend,
        report_fn=report_fn,
        save_path=save_path,
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
