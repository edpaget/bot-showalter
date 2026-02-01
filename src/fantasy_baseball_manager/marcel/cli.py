from collections.abc import Callable
from datetime import datetime
from typing import Annotated

import typer

from fantasy_baseball_manager.marcel.batting import project_batters
from fantasy_baseball_manager.marcel.data_source import PybaseballDataSource, StatsDataSource
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.marcel.pitching import project_pitchers

BATTING_SORT_FIELDS: dict[str, Callable[[BattingProjection], float]] = {
    "hr": lambda p: p.hr,
    "pa": lambda p: p.pa,
    "h": lambda p: p.h,
    "sb": lambda p: p.sb,
    "bb": lambda p: p.bb,
    "so": lambda p: p.so,
    "avg": lambda p: p.h / p.ab if p.ab > 0 else 0,
    "obp": lambda p: (p.h + p.bb + p.hbp) / p.pa if p.pa > 0 else 0,
}

PITCHING_SORT_FIELDS: dict[str, Callable[[PitchingProjection], float]] = {
    "era": lambda p: -p.era,  # lower is better
    "so": lambda p: p.so,
    "ip": lambda p: p.ip,
    "whip": lambda p: -p.whip,  # lower is better
    "hr": lambda p: p.hr,
    "bb": lambda p: p.bb,
}

# Module-level factory for dependency injection in tests
_data_source_factory: Callable[[], StatsDataSource] = PybaseballDataSource


def set_data_source_factory(factory: Callable[[], StatsDataSource]) -> None:
    global _data_source_factory
    _data_source_factory = factory


def format_batting_table(projections: list[BattingProjection], top: int) -> str:
    lines: list[str] = []
    lines.append(f"Top {top} projected batters ({len(projections)} total):")
    lines.append(f"{'Name':<25} {'Age':>3} {'PA':>6} {'HR':>5} {'AVG':>6} {'OBP':>6} {'SB':>5}")
    lines.append("-" * 60)
    for p in projections[:top]:
        avg = p.h / p.ab if p.ab > 0 else 0
        obp = (p.h + p.bb + p.hbp) / p.pa if p.pa > 0 else 0
        lines.append(f"{p.name:<25} {p.age:>3} {p.pa:>6.0f} {p.hr:>5.1f} {avg:>6.3f} {obp:>6.3f} {p.sb:>5.1f}")
    return "\n".join(lines)


def format_pitching_table(projections: list[PitchingProjection], top: int) -> str:
    lines: list[str] = []
    lines.append(f"Top {top} projected pitchers ({len(projections)} total):")
    lines.append(f"{'Name':<25} {'Age':>3} {'IP':>6} {'ERA':>5} {'WHIP':>5} {'SO':>5} {'HR':>4}")
    lines.append("-" * 58)
    for p in projections[:top]:
        lines.append(f"{p.name:<25} {p.age:>3} {p.ip:>6.1f} {p.era:>5.2f} {p.whip:>5.3f} {p.so:>5.1f} {p.hr:>4.1f}")
    return "\n".join(lines)


def marcel(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    batting: Annotated[bool, typer.Option("--batting", help="Show only batting projections.")] = False,
    pitching: Annotated[bool, typer.Option("--pitching", help="Show only pitching projections.")] = False,
    top: Annotated[int, typer.Option(help="Number of players to display.")] = 20,
    sort_by: Annotated[str | None, typer.Option(help="Stat to sort by (e.g. hr, so, era).")] = None,
) -> None:
    """Generate MARCEL projections for the given year."""
    if year is None:
        year = datetime.now().year

    show_batting = not pitching or batting
    show_pitching = not batting or pitching

    typer.echo(f"MARCEL projections for {year}")
    typer.echo(f"Using data from {year - 3}-{year - 1}\n")

    data_source = _data_source_factory()

    if show_batting:
        batting_sort = sort_by or "hr"
        if batting_sort not in BATTING_SORT_FIELDS:
            typer.echo(f"Unknown batting sort field: {batting_sort}", err=True)
            raise typer.Exit(code=1)
        projections = project_batters(data_source, year)
        projections.sort(key=BATTING_SORT_FIELDS[batting_sort], reverse=True)
        typer.echo(format_batting_table(projections, top))

    if show_batting and show_pitching:
        typer.echo()

    if show_pitching:
        pitching_sort = sort_by or "so"
        if pitching_sort not in PITCHING_SORT_FIELDS:
            typer.echo(f"Unknown pitching sort field: {pitching_sort}", err=True)
            raise typer.Exit(code=1)
        projections = project_pitchers(data_source, year)
        projections.sort(key=PITCHING_SORT_FIELDS[pitching_sort], reverse=True)
        typer.echo(format_pitching_table(projections, top))
