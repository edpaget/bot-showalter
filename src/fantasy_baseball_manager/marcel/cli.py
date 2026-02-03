from collections.abc import Callable
from datetime import datetime
from typing import Annotated

import typer

from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.marcel.models import BattingProjection, PitchingProjection
from fantasy_baseball_manager.pipeline.builder import PipelineBuilder
from fantasy_baseball_manager.pipeline.engine import ProjectionPipeline
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.services import get_container, set_container

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
    "format_batting_table",
    "format_pitching_table",
    "marcel",
    "set_container",
]


def format_batting_table(
    projections: list[BattingProjection],
    top: int,
    title: str = "projected batters",
) -> str:
    lines: list[str] = []
    lines.append(f"Top {top} {title} ({len(projections)} total):")
    lines.append(f"{'Name':<25} {'Age':>3} {'PA':>6} {'HR':>5} {'R':>5} {'RBI':>5} {'AVG':>6} {'OBP':>6} {'SB':>5}")
    lines.append("-" * 72)
    for p in projections[:top]:
        avg = p.h / p.ab if p.ab > 0 else 0
        obp = (p.h + p.bb + p.hbp) / p.pa if p.pa > 0 else 0
        lines.append(
            f"{p.name:<25} {p.age:>3} {p.pa:>6.0f} {p.hr:>5.1f} {p.r:>5.1f} {p.rbi:>5.1f} {avg:>6.3f} {obp:>6.3f} {p.sb:>5.1f}"
        )
    return "\n".join(lines)


def format_pitching_table(
    projections: list[PitchingProjection],
    top: int,
    title: str = "projected pitchers",
) -> str:
    lines: list[str] = []
    lines.append(f"Top {top} {title} ({len(projections)} total):")
    lines.append(f"{'Name':<25} {'Age':>3} {'IP':>6} {'ERA':>5} {'WHIP':>5} {'SO':>5} {'W':>4} {'NSVH':>5} {'HR':>4}")
    lines.append("-" * 68)
    for p in projections[:top]:
        lines.append(
            f"{p.name:<25} {p.age:>3} {p.ip:>6.1f} {p.era:>5.2f} {p.whip:>5.3f} {p.so:>5.1f} {p.w:>4.1f} {p.nsvh:>5.1f} {p.hr:>4.1f}"
        )
    return "\n".join(lines)


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
    pitcher_norm: Annotated[bool, typer.Option("--pitcher-norm", help="Enable pitcher BABIP/LOB normalization.")] = False,
    statcast: Annotated[bool, typer.Option("--statcast", help="Enable Statcast batter blend.")] = False,
    batter_babip: Annotated[bool, typer.Option("--batter-babip", help="Enable batter BABIP adjustment.")] = False,
    pitcher_statcast: Annotated[bool, typer.Option("--pitcher-statcast", help="Enable pitcher Statcast blend.")] = False,
) -> None:
    """Generate projections for the given year."""
    if year is None:
        year = datetime.now().year

    show_batting = not pitching or batting
    show_pitching = not batting or pitching

    pipeline = _build_pipeline_from_flags(
        engine, park_factors, pitcher_norm, statcast, batter_babip, pitcher_statcast,
    )

    typer.echo(f"{pipeline.name.upper()} projections for {year}")
    typer.echo(f"Using data from {year - pipeline.years_back}-{year - 1}\n")

    data_source = get_container().data_source

    if show_batting:
        batting_sort = sort_by or "hr"
        if batting_sort not in BATTING_SORT_FIELDS:
            typer.echo(f"Unknown batting sort field: {batting_sort}", err=True)
            raise typer.Exit(code=1)
        projections = pipeline.project_batters(data_source, year)
        projections.sort(key=BATTING_SORT_FIELDS[batting_sort], reverse=True)
        typer.echo(format_batting_table(projections, top))

    if show_batting and show_pitching:
        typer.echo()

    if show_pitching:
        pitching_sort = sort_by or "so"
        if pitching_sort not in PITCHING_SORT_FIELDS:
            typer.echo(f"Unknown pitching sort field: {pitching_sort}", err=True)
            raise typer.Exit(code=1)
        projections = pipeline.project_pitchers(data_source, year)
        projections.sort(key=PITCHING_SORT_FIELDS[pitching_sort], reverse=True)
        typer.echo(format_pitching_table(projections, top))
