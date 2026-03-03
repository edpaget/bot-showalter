from typing import Annotated

import typer

from fantasy_baseball_manager.cli._helpers import parse_system_version
from fantasy_baseball_manager.cli._output import print_error, print_error_decomposition_report
from fantasy_baseball_manager.cli.factory import build_residuals_context

residuals_app = typer.Typer(name="residuals", help="Residual analysis tools")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


@residuals_app.command("worst-misses")
def worst_misses(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    target: Annotated[str, typer.Option("--target", help="Target stat (e.g. slg, era)")],
    top: Annotated[int, typer.Option("--top", help="Number of worst misses")] = 20,
    direction: Annotated[str | None, typer.Option("--direction", help="over or under")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show the worst prediction misses for a given stat."""
    sys_name, version = parse_system_version(system)

    if direction is not None and direction not in ("over", "under"):
        print_error(f"invalid direction '{direction}', expected 'over' or 'under'")
        raise typer.Exit(code=1)

    with build_residuals_context(data_dir) as ctx:
        report = ctx.analyzer.analyze(
            sys_name,
            version,
            season,
            target,
            player_type,
            top_n=top,
            direction=direction,
        )

    if not report.top_misses:
        print_error("no matching projections/actuals found")
        raise typer.Exit(code=1)

    print_error_decomposition_report(report)
