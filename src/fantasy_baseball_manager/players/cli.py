import typer

from fantasy_baseball_manager.draft.cli import draft_rank, draft_simulate
from fantasy_baseball_manager.marcel.cli import marcel
from fantasy_baseball_manager.ros.cli import ros_project
from fantasy_baseball_manager.valuation.cli import valuate

players_app = typer.Typer(help="Player projection and valuation commands.")
players_app.command(name="draft-rank")(draft_rank)
players_app.command(name="draft-simulate")(draft_simulate)
players_app.command(name="project")(marcel)
players_app.command(name="ros-project")(ros_project)
players_app.command(name="valuate")(valuate)
