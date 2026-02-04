from collections.abc import Callable
from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.pipeline.builder import PipelineBuilder
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.services import get_container, set_container

console = Console()

BATTING_SORT_FIELDS: dict[str, Callable[[BattingProjection], float]] = {
    "hr": lambda p: p.hr,
    "r": lambda p: p.r,
    "rbi": lambda p: p.rbi,
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
    "w": lambda p: p.w,
    "nsvh": lambda p: p.nsvh,
}

__all__ = [
    "BATTING_SORT_FIELDS",
    "PITCHING_SORT_FIELDS",
    "marcel",
    "print_batting_table",
    "print_pitching_table",
    "set_container",
]


def print_batting_table(
    projections: list[BattingProjection],
    top: int,
    title: str = "projected batters",
) -> None:
    table = Table(title=f"Top {top} {title} ({len(projections)} total)")
    table.add_column("Name")
    table.add_column("Age", justify="right")
    table.add_column("PA", justify="right")
    table.add_column("HR", justify="right")
    table.add_column("R", justify="right")
    table.add_column("RBI", justify="right")
    table.add_column("AVG", justify="right")
    table.add_column("OBP", justify="right")
    table.add_column("SB", justify="right")

    for p in projections[:top]:
        avg = p.h / p.ab if p.ab > 0 else 0
        obp = (p.h + p.bb + p.hbp) / p.pa if p.pa > 0 else 0
        table.add_row(
            p.name,
            str(p.age),
            f"{p.pa:.0f}",
            f"{p.hr:.1f}",
            f"{p.r:.1f}",
            f"{p.rbi:.1f}",
            f"{avg:.3f}",
            f"{obp:.3f}",
            f"{p.sb:.1f}",
        )
    console.print(table)


def print_pitching_table(
    projections: list[PitchingProjection],
    top: int,
    title: str = "projected pitchers",
) -> None:
    table = Table(title=f"Top {top} {title} ({len(projections)} total)")
    table.add_column("Name")
    table.add_column("Age", justify="right")
    table.add_column("IP", justify="right")
    table.add_column("ERA", justify="right")
    table.add_column("WHIP", justify="right")
    table.add_column("SO", justify="right")
    table.add_column("W", justify="right")
    table.add_column("NSVH", justify="right")
    table.add_column("HR", justify="right")

    for p in projections[:top]:
        table.add_row(
            p.name,
            str(p.age),
            f"{p.ip:.1f}",
            f"{p.era:.2f}",
            f"{p.whip:.3f}",
            f"{p.so:.1f}",
            f"{p.w:.1f}",
            f"{p.nsvh:.1f}",
            f"{p.hr:.1f}",
        )
    console.print(table)


def _build_pipeline_from_flags(
    engine: str,
    park_factors: bool,
    pitcher_norm: bool,
    statcast: bool,
    batter_babip: bool,
    pitcher_statcast: bool,
) -> ProjectionPipeline:
    """Build a pipeline from a preset name or ad-hoc feature flags."""
    has_flags = any([park_factors, pitcher_norm, statcast, batter_babip, pitcher_statcast])

    if has_flags and engine != DEFAULT_ENGINE:
        typer.echo("Cannot combine --engine with feature flags (--park-factors, etc.).", err=True)
        raise typer.Exit(code=1)

    if has_flags:
        builder = PipelineBuilder("custom")
        if park_factors:
            builder = builder.with_park_factors()
        if pitcher_norm:
            builder = builder.with_pitcher_normalization()
        if statcast:
            builder = builder.with_statcast()
        if batter_babip:
            builder = builder.with_batter_babip()
        if pitcher_statcast:
            builder = builder.with_pitcher_statcast()
        return builder.build()

    validate_engine(engine)
    return PIPELINES[engine]()


def marcel(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    batting: Annotated[bool, typer.Option("--batting", help="Show only batting projections.")] = False,
    pitching: Annotated[bool, typer.Option("--pitching", help="Show only pitching projections.")] = False,
    top: Annotated[int, typer.Option(help="Number of players to display.")] = 20,
    sort_by: Annotated[str | None, typer.Option(help="Stat to sort by (e.g. hr, so, era).")] = None,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
    park_factors: Annotated[bool, typer.Option("--park-factors", help="Enable park factor adjustment.")] = False,
    pitcher_norm: Annotated[
        bool, typer.Option("--pitcher-norm", help="Enable pitcher BABIP/LOB normalization.")
    ] = False,
    statcast: Annotated[bool, typer.Option("--statcast", help="Enable Statcast batter blend.")] = False,
    batter_babip: Annotated[bool, typer.Option("--batter-babip", help="Enable batter BABIP adjustment.")] = False,
    pitcher_statcast: Annotated[
        bool, typer.Option("--pitcher-statcast", help="Enable pitcher Statcast blend.")
    ] = False,
) -> None:
    """Generate projections for the given year."""
    if year is None:
        year = datetime.now().year

    show_batting = not pitching or batting
    show_pitching = not batting or pitching

    pipeline = _build_pipeline_from_flags(
        engine,
        park_factors,
        pitcher_norm,
        statcast,
        batter_babip,
        pitcher_statcast,
    )

    console.print(f"[bold]{pipeline.name.upper()} projections for {year}[/bold]")
    console.print(f"Using data from {year - pipeline.years_back}-{year - 1}\n")

    data_source = get_container().data_source

    if show_batting:
        batting_sort = sort_by or "hr"
        if batting_sort not in BATTING_SORT_FIELDS:
            typer.echo(f"Unknown batting sort field: {batting_sort}", err=True)
            raise typer.Exit(code=1)
        projections = pipeline.project_batters(data_source, year)
        projections.sort(key=BATTING_SORT_FIELDS[batting_sort], reverse=True)
        print_batting_table(projections, top)

    if show_batting and show_pitching:
        console.print()

    if show_pitching:
        pitching_sort = sort_by or "so"
        if pitching_sort not in PITCHING_SORT_FIELDS:
            typer.echo(f"Unknown pitching sort field: {pitching_sort}", err=True)
            raise typer.Exit(code=1)
        projections = pipeline.project_pitchers(data_source, year)
        projections.sort(key=PITCHING_SORT_FIELDS[pitching_sort], reverse=True)
        print_pitching_table(projections, top)
