import typer

from fantasy_baseball_manager.marcel.cli import marcel

project_app = typer.Typer(help="Projection commands.")
project_app.command()(marcel)

app = typer.Typer(help="Fantasy baseball manager.")
app.add_typer(project_app, name="project")
