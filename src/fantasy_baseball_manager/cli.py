import logging

import typer

from fantasy_baseball_manager.evaluation.cli import evaluate_cmd
from fantasy_baseball_manager.keeper.cli import keeper_app
from fantasy_baseball_manager.players.cli import players_app
from fantasy_baseball_manager.teams.cli import teams_app

app = typer.Typer(help="Fantasy baseball manager.")
app.add_typer(players_app, name="players")
app.add_typer(teams_app, name="teams")
app.add_typer(keeper_app, name="keeper")
app.command(name="evaluate")(evaluate_cmd)


@app.callback()
def main_callback(
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity (-v debug, -vvv urllib)."),
) -> None:
    if verbose >= 1:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
        # Suppress noisy HTTP/urllib loggers unless -vvv
        if verbose < 3:
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("http.client").setLevel(logging.WARNING)
            logging.getLogger("urllib.request").setLevel(logging.WARNING)
