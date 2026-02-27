from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import console
from fantasy_baseball_manager.cli.factory import build_compute_container

compute_app = typer.Typer(name="compute", help="Compute derived data from ingested stats")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]

_MILB_COMPUTE_LEVELS = ["AAA", "AA", "A+", "A", "ROK"]


@compute_app.command("league-env")
def compute_league_env(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    level: Annotated[list[str] | None, typer.Option("--level", help="Level(s): AAA, AA, A+, A, ROK")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Compute league environment aggregates from minor league batting stats."""
    levels = level if level else _MILB_COMPUTE_LEVELS
    with build_compute_container(data_dir) as container:
        for yr in season:
            for lvl in levels:
                count = container.league_environment_service.compute_for_season_level(yr, lvl)
                container.conn.commit()
                console.print(f"  {lvl} {yr}: {count} league(s) computed")
    console.print("[bold green]Done.[/bold green]")
