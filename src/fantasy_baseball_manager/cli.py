import typer

from fantasy_baseball_manager.evaluation.cli import evaluate_cmd
from fantasy_baseball_manager.players.cli import players_app
from fantasy_baseball_manager.teams.cli import teams_app

app = typer.Typer(help="Fantasy baseball manager.")
app.add_typer(players_app, name="players")
app.add_typer(teams_app, name="teams")
app.command(name="evaluate")(evaluate_cmd)
