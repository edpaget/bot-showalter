import typer

from fantasy_baseball_manager.league.cli import compare, projections

teams_app = typer.Typer(help="Team roster and comparison commands.")
teams_app.command(name="roster")(projections)
teams_app.command(name="compare")(compare)
