from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import print_player_projections, print_pt_sources, print_system_summaries
from fantasy_baseball_manager.cli.factory import build_projections_context

projections_app = typer.Typer(name="projections", help="Look up and explore projection systems")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


@projections_app.command("lookup")
def projections_lookup(
    player_name: Annotated[str, typer.Argument(help="Player name ('Last' or 'Last, First')")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str | None, typer.Option("--system", help="Filter by system")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Look up a player's projections across systems."""
    with build_projections_context(data_dir) as ctx:
        results = ctx.lookup_service.lookup(player_name, season, system=system)
    print_player_projections(results)


@projections_app.command("systems")
def projections_systems(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """List available projection systems for a season."""
    with build_projections_context(data_dir) as ctx:
        summaries = ctx.lookup_service.list_systems(season)
    print_system_summaries(summaries)


@projections_app.command("pt-sources")
def projections_pt_sources(  # pragma: no cover
    season: Annotated[int, typer.Option("--season", help="Season year")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """List projection systems that provide playing-time (PA/IP) data."""
    with build_projections_context(data_dir) as ctx:
        sources = ctx.lookup_service.list_pt_sources(season)
    print_pt_sources(sources)
