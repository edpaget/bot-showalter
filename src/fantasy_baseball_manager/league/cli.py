import io
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fantasy_baseball_manager.config import load_league_settings
from fantasy_baseball_manager.engines import DEFAULT_ENGINE, DEFAULT_METHOD, validate_engine, validate_method
from fantasy_baseball_manager.league.models import TeamProjection
from fantasy_baseball_manager.league.projections import match_projections
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.services import cli_context, get_container, set_container
from fantasy_baseball_manager.valuation.models import LeagueSettings, StatCategory

logger = logging.getLogger(__name__)

COMPARE_SORT_FIELDS: dict[str, Callable[[TeamProjection], float]] = {
    "total_hr": lambda t: t.total_hr,
    "total_sb": lambda t: t.total_sb,
    "total_h": lambda t: t.total_h,
    "total_pa": lambda t: t.total_pa,
    "team_avg": lambda t: t.team_avg,
    "team_obp": lambda t: t.team_obp,
    "total_r": lambda t: t.total_r,
    "total_rbi": lambda t: t.total_rbi,
    "total_ip": lambda t: t.total_ip,
    "total_so": lambda t: t.total_so,
    "total_w": lambda t: t.total_w,
    "total_nsvh": lambda t: t.total_nsvh,
    "team_era": lambda t: -t.team_era,  # lower is better
    "team_whip": lambda t: -t.team_whip,  # lower is better
}

# Per-player column specs: (header, format_spec, extractor)
_BATTER_COLUMNS: dict[StatCategory, tuple[str, str, Callable[[BattingProjection], float]]] = {
    StatCategory.HR: ("HR", ".1f", lambda bp: bp.hr),
    StatCategory.R: ("R", ".1f", lambda bp: bp.r),
    StatCategory.RBI: ("RBI", ".1f", lambda bp: bp.rbi),
    StatCategory.SB: ("SB", ".1f", lambda bp: bp.sb),
    StatCategory.OBP: ("OBP", ".3f", lambda bp: (bp.h + bp.bb + bp.hbp) / bp.pa if bp.pa > 0 else 0),
}

_PITCHER_COLUMNS: dict[StatCategory, tuple[str, str, Callable[[PitchingProjection], float]]] = {
    StatCategory.W: ("W", ".1f", lambda pp: pp.w),
    StatCategory.K: ("K", ".1f", lambda pp: pp.so),
    StatCategory.ERA: ("ERA", ".2f", lambda pp: pp.era),
    StatCategory.WHIP: ("WHIP", ".3f", lambda pp: pp.whip),
    StatCategory.NSVH: ("NSVH", ".1f", lambda pp: pp.nsvh),
}

# Team aggregate column specs: (header, format_spec, extractor)
_TEAM_BATTING_COLUMNS: dict[StatCategory, tuple[str, str, Callable[[TeamProjection], float]]] = {
    StatCategory.HR: ("HR", ".0f", lambda t: t.total_hr),
    StatCategory.R: ("R", ".0f", lambda t: t.total_r),
    StatCategory.RBI: ("RBI", ".0f", lambda t: t.total_rbi),
    StatCategory.SB: ("SB", ".0f", lambda t: t.total_sb),
    StatCategory.OBP: ("OBP", ".3f", lambda t: t.team_obp),
}

_TEAM_PITCHING_COLUMNS: dict[StatCategory, tuple[str, str, Callable[[TeamProjection], float]]] = {
    StatCategory.W: ("W", ".0f", lambda t: t.total_w),
    StatCategory.K: ("K", ".0f", lambda t: t.total_so),
    StatCategory.ERA: ("ERA", ".2f", lambda t: t.team_era),
    StatCategory.WHIP: ("WHIP", ".3f", lambda t: t.team_whip),
    StatCategory.NSVH: ("NSVH", ".0f", lambda t: t.total_nsvh),
}

console = Console()

__all__ = ["compare", "format_compare_table", "format_team_projections", "projections", "set_container"]


def print_team_projections(
    team_projections: list[TeamProjection],
    league_settings: LeagueSettings,
) -> None:
    """Print team projections using rich tables."""
    bat_cols = [(cat, _BATTER_COLUMNS[cat]) for cat in league_settings.batting_categories if cat in _BATTER_COLUMNS]
    pit_cols = [(cat, _PITCHER_COLUMNS[cat]) for cat in league_settings.pitching_categories if cat in _PITCHER_COLUMNS]

    for team in team_projections:
        console.print(f"\n[bold]{team.team_name}[/bold] ({team.team_key})")

        batters = [p for p in team.players if p.roster_player.position_type == "B"]
        pitchers = [p for p in team.players if p.roster_player.position_type == "P"]

        if batters:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Batters")
            table.add_column("PA", justify="right")
            for _, (header, _, _) in bat_cols:
                table.add_column(header, justify="right")

            for pm in batters:
                if pm.batting_projection is not None:
                    bp = pm.batting_projection
                    row = [pm.roster_player.name, f"{bp.pa:.0f}"]
                    for _, (_, fmt, extract) in bat_cols:
                        row.append(f"{extract(bp):{fmt}}")
                    table.add_row(*row)
                else:
                    row = [pm.roster_player.name, "--"]
                    row.extend(["--"] * len(bat_cols))
                    table.add_row(*row)
            console.print(table)

        if pitchers:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Pitchers")
            table.add_column("IP", justify="right")
            for _, (header, _, _) in pit_cols:
                table.add_column(header, justify="right")

            for pm in pitchers:
                if pm.pitching_projection is not None:
                    pp = pm.pitching_projection
                    row = [pm.roster_player.name, f"{pp.ip:.1f}"]
                    for _, (_, fmt, extract) in pit_cols:
                        row.append(f"{extract(pp):{fmt}}")
                    table.add_row(*row)
                else:
                    row = [pm.roster_player.name, "--"]
                    row.extend(["--"] * len(pit_cols))
                    table.add_row(*row)
            console.print(table)

        if team.unmatched_count > 0:
            console.print(
                f"[yellow]Warning: {team.unmatched_count} player(s) could not be matched to projections[/yellow]"
            )


def print_compare_table(
    team_projections: list[TeamProjection],
    league_settings: LeagueSettings,
) -> None:
    """Print team comparison table using rich."""
    bat_cols = [
        (cat, _TEAM_BATTING_COLUMNS[cat]) for cat in league_settings.batting_categories if cat in _TEAM_BATTING_COLUMNS
    ]
    pit_cols = [
        (cat, _TEAM_PITCHING_COLUMNS[cat])
        for cat in league_settings.pitching_categories
        if cat in _TEAM_PITCHING_COLUMNS
    ]

    table = Table(show_header=True, header_style="bold")
    table.add_column("Team")
    for _, (header, _, _) in bat_cols:
        table.add_column(header, justify="right")
    for _, (header, _, _) in pit_cols:
        table.add_column(header, justify="right")
    table.add_column("?", justify="right")

    for t in team_projections:
        row: list[str] = [t.team_name]
        for _, (_, fmt, extract) in bat_cols:
            row.append(f"{extract(t):{fmt}}")
        for _, (_, fmt, extract) in pit_cols:
            row.append(f"{extract(t):{fmt}}")
        row.append(str(t.unmatched_count))
        table.add_row(*row)

    console.print(table)


def format_team_projections(
    team_projections: list[TeamProjection],
    league_settings: LeagueSettings,
) -> str:
    """Format team projections as a string.

    Returns a string representation of the team projections table,
    useful for testing or non-interactive output.
    """
    bat_cols = [(cat, _BATTER_COLUMNS[cat]) for cat in league_settings.batting_categories if cat in _BATTER_COLUMNS]
    pit_cols = [(cat, _PITCHER_COLUMNS[cat]) for cat in league_settings.pitching_categories if cat in _PITCHER_COLUMNS]

    string_io = io.StringIO()
    string_console = Console(file=string_io, force_terminal=True, width=120)

    for team in team_projections:
        string_console.print(f"\n[bold]{team.team_name}[/bold] ({team.team_key})")

        batters = [p for p in team.players if p.roster_player.position_type == "B"]
        pitchers = [p for p in team.players if p.roster_player.position_type == "P"]

        if batters:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Batters")
            table.add_column("PA", justify="right")
            for _, (header, _, _) in bat_cols:
                table.add_column(header, justify="right")

            for pm in batters:
                if pm.batting_projection is not None:
                    bp = pm.batting_projection
                    row = [pm.roster_player.name, f"{bp.pa:.0f}"]
                    for _, (_, fmt, extract) in bat_cols:
                        row.append(f"{extract(bp):{fmt}}")
                    table.add_row(*row)
                else:
                    row = [pm.roster_player.name, "--"]
                    row.extend(["--"] * len(bat_cols))
                    table.add_row(*row)
            string_console.print(table)

        if pitchers:
            table = Table(show_header=True, header_style="bold")
            table.add_column("Pitchers")
            table.add_column("IP", justify="right")
            for _, (header, _, _) in pit_cols:
                table.add_column(header, justify="right")

            for pm in pitchers:
                if pm.pitching_projection is not None:
                    pp = pm.pitching_projection
                    row = [pm.roster_player.name, f"{pp.ip:.1f}"]
                    for _, (_, fmt, extract) in pit_cols:
                        row.append(f"{extract(pp):{fmt}}")
                    table.add_row(*row)
                else:
                    row = [pm.roster_player.name, "--"]
                    row.extend(["--"] * len(pit_cols))
                    table.add_row(*row)
            string_console.print(table)

        if team.unmatched_count > 0:
            string_console.print(
                f"[yellow]Warning: {team.unmatched_count} player(s) could not be matched to projections[/yellow]"
            )

    return string_io.getvalue()


def format_compare_table(
    team_projections: list[TeamProjection],
    league_settings: LeagueSettings,
) -> str:
    """Format team comparison table as a string.

    Returns a string representation of the comparison table,
    useful for testing or non-interactive output.
    """
    bat_cols = [
        (cat, _TEAM_BATTING_COLUMNS[cat]) for cat in league_settings.batting_categories if cat in _TEAM_BATTING_COLUMNS
    ]
    pit_cols = [
        (cat, _TEAM_PITCHING_COLUMNS[cat])
        for cat in league_settings.pitching_categories
        if cat in _TEAM_PITCHING_COLUMNS
    ]

    string_io = io.StringIO()
    string_console = Console(file=string_io, force_terminal=True, width=120)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Team")
    for _, (header, _, _) in bat_cols:
        table.add_column(header, justify="right")
    for _, (header, _, _) in pit_cols:
        table.add_column(header, justify="right")
    table.add_column("?", justify="right")

    for t in team_projections:
        row: list[str] = [t.team_name]
        for _, (_, fmt, extract) in bat_cols:
            row.append(f"{extract(t):{fmt}}")
        for _, (_, fmt, extract) in pit_cols:
            row.append(f"{extract(t):{fmt}}")
        row.append(str(t.unmatched_count))
        table.add_row(*row)

    string_console.print(table)
    return string_io.getvalue()


def _load_team_projections(year: int, engine: str = DEFAULT_ENGINE) -> list[TeamProjection]:
    container = get_container()
    if container.config.no_cache:
        container.invalidate_caches()

    pipeline = PIPELINES[engine]()
    rosters = container.roster_source.fetch_rosters()
    batting = pipeline.project_batters(container.batting_source, container.team_batting_source, year)
    pitching = pipeline.project_pitchers(container.pitching_source, container.team_pitching_source, year)

    return match_projections(rosters, batting, pitching, container.id_mapper)


def projections(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    sort_by: Annotated[str, typer.Option(help="Stat to sort teams by.")] = "total_hr",
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data from Yahoo API.")
    ] = False,
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Show projections for all rostered players in the league."""
    with cli_context(league_id=league_id, season=season, no_cache=no_cache):
        validate_engine(engine)

        if year is None:
            year = datetime.now().year

        if sort_by not in COMPARE_SORT_FIELDS:
            typer.echo(f"Unknown sort field: {sort_by}", err=True)
            raise typer.Exit(code=1)

        league_settings = load_league_settings()
        console.print(f"[bold]League projections for {year}[/bold]\n")

        team_projections = _load_team_projections(year, engine=engine)
        team_projections.sort(key=COMPARE_SORT_FIELDS[sort_by], reverse=True)

        print_team_projections(team_projections, league_settings)


def compare(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    sort_by: Annotated[str, typer.Option(help="Stat to sort by.")] = "total_hr",
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    method: Annotated[str, typer.Option(help="Valuation method to use.")] = DEFAULT_METHOD,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass cache and fetch fresh data from Yahoo API.")
    ] = False,
    league_id: Annotated[str | None, typer.Option("--league-id", help="Override league ID from config.")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Override season from config.")] = None,
) -> None:
    """Compare aggregate projected stats across all teams in the league."""
    with cli_context(league_id=league_id, season=season, no_cache=no_cache):
        validate_engine(engine)
        validate_method(method)

        if year is None:
            year = datetime.now().year

        if sort_by not in COMPARE_SORT_FIELDS:
            typer.echo(f"Unknown sort field: {sort_by}", err=True)
            raise typer.Exit(code=1)

        league_settings = load_league_settings()
        console.print(f"[bold]League comparison for {year}[/bold]\n")

        team_projections = _load_team_projections(year, engine=engine)
        team_projections.sort(key=COMPARE_SORT_FIELDS[sort_by], reverse=True)

        print_compare_table(team_projections, league_settings)
