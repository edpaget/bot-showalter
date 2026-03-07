from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.cli._output import console
from fantasy_baseball_manager.cli.factory import build_compute_container
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.services import PlayerEligibilityService, compute_actual_valuations

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


@compute_app.command("actual-valuations")
def compute_actual_valuations_cmd(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "h2h",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Compute ZAR valuations from end-of-season actual stats."""
    league = load_league(league_name, Path.cwd())
    conn = create_connection(Path(data_dir) / "fbm.db")
    try:
        container = AnalysisContainer(SingleConnectionProvider(conn))
        eligibility = PlayerEligibilityService(
            container.position_appearance_repo,
            pitching_stats_repo=container.pitching_stats_repo,
        )
        for yr in season:
            valuations = compute_actual_valuations(
                season=yr,
                league=league,
                batting_repo=container.batting_stats_repo,
                pitching_repo=container.pitching_stats_repo,
                position_repo=container.position_appearance_repo,
                valuation_repo=container.valuation_repo,
                eligibility_provider=eligibility,
            )
            conn.commit()
            console.print(f"  {yr}: {len(valuations)} actual valuations computed")
    finally:
        conn.close()
    console.print("[bold green]Done.[/bold green]")
