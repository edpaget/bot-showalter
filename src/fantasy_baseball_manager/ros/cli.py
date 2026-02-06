from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console

from fantasy_baseball_manager.engines import DEFAULT_ENGINE, validate_engine
from fantasy_baseball_manager.marcel.cli import (
    BATTING_SORT_FIELDS,
    PITCHING_SORT_FIELDS,
    print_batting_table,
    print_pitching_table,
)
from fantasy_baseball_manager.pipeline.presets import PIPELINES
from fantasy_baseball_manager.ros.projector import ROSProjector
from fantasy_baseball_manager.services import get_container, set_container

console = Console()

__all__ = ["ros_project", "set_container"]


def ros_project(
    year: Annotated[int | None, typer.Argument(help="Projection year (default: current year).")] = None,
    batting: Annotated[bool, typer.Option("--batting", help="Show only batting projections.")] = False,
    pitching: Annotated[bool, typer.Option("--pitching", help="Show only pitching projections.")] = False,
    top: Annotated[int, typer.Option(help="Number of players to display.")] = 20,
    sort_by: Annotated[str | None, typer.Option(help="Stat to sort by (e.g. hr, so, era).")] = None,
    engine: Annotated[str, typer.Option(help="Projection engine to use.")] = DEFAULT_ENGINE,
) -> None:
    """Generate rest-of-season projections by blending pre-season projections with current-year actuals."""
    validate_engine(engine)

    if year is None:
        year = datetime.now().year

    show_batting = not pitching or batting
    show_pitching = not batting or pitching

    pipeline = PIPELINES[engine]()
    container = get_container()
    projector = ROSProjector(
        pipeline=pipeline,
        batting_source=container.batting_source,
        team_batting_source=container.team_batting_source,
        pitching_source=container.pitching_source,
        team_pitching_source=container.team_pitching_source,
        blender=container.blender,
    )

    console.print(f"[bold]ROS projections for {year} (engine: {engine})[/bold]")
    console.print(f"Using pre-season prior from {year - pipeline.years_back}-{year - 1}\n")

    if show_batting:
        batting_sort = sort_by or "hr"
        if batting_sort not in BATTING_SORT_FIELDS:
            typer.echo(f"Unknown batting sort field: {batting_sort}", err=True)
            raise typer.Exit(code=1)
        projections = projector.project_batters(year)
        projections.sort(key=BATTING_SORT_FIELDS[batting_sort], reverse=True)
        print_batting_table(projections, top, title="ROS batters")

    if show_batting and show_pitching:
        console.print()

    if show_pitching:
        pitching_sort = sort_by or "so"
        if pitching_sort not in PITCHING_SORT_FIELDS:
            typer.echo(f"Unknown pitching sort field: {pitching_sort}", err=True)
            raise typer.Exit(code=1)
        projections = projector.project_pitchers(year)
        projections.sort(key=PITCHING_SORT_FIELDS[pitching_sort], reverse=True)
        print_pitching_table(projections, top, title="ROS pitchers")
