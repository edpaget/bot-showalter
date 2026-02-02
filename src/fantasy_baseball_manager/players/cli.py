import typer

from fantasy_baseball_manager.draft.cli import draft_rank
from fantasy_baseball_manager.marcel.cli import marcel
from fantasy_baseball_manager.valuation.cli import valuate

players_app = typer.Typer(help="Player projection and valuation commands.")
players_app.command(name="draft-rank")(draft_rank)
players_app.command(name="project")(marcel)
players_app.command(name="valuate")(valuate)
