import csv
from pathlib import Path  # noqa: TC003 — Typer evaluates annotations at runtime
from typing import Annotated

import typer

from fantasy_baseball_manager.cli.factory import build_keeper_context
from fantasy_baseball_manager.domain import Err, Ok
from fantasy_baseball_manager.ingest import import_keeper_costs
from fantasy_baseball_manager.services import set_keeper_cost

keeper_app = typer.Typer(name="keeper", help="Keeper league cost management")


@keeper_app.command("import")
def import_cmd(
    csv_path: Annotated[Path, typer.Argument(help="Path to keeper costs CSV file")],
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    source: Annotated[str, typer.Option(help="Cost source type")] = "auction",
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Import keeper costs from a CSV file."""
    if not csv_path.exists():
        typer.echo(f"Error: file not found: {csv_path}", err=True)
        raise typer.Exit(code=1)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with build_keeper_context(data_dir) as ctx:
        players = ctx.player_repo.all()
        result = import_keeper_costs(
            rows, ctx.keeper_repo, players, season=season, league=league, default_source=source
        )
        ctx.conn.commit()

    typer.echo(f"Loaded {result.loaded} keeper costs, skipped {result.skipped}")
    if result.unmatched:
        typer.echo(f"Unmatched players ({len(result.unmatched)}): {', '.join(result.unmatched)}")


@keeper_app.command("set")
def set_cmd(
    player_name: Annotated[str, typer.Argument(help="Player name to search for")],
    cost: Annotated[float, typer.Option(help="Keeper cost")],
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    years: Annotated[int, typer.Option(help="Years remaining on contract")] = 1,
    source: Annotated[str, typer.Option(help="Cost source type")] = "auction",
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Set a keeper cost for a single player."""
    with build_keeper_context(data_dir) as ctx:
        result = set_keeper_cost(player_name, cost, season, league, ctx.player_repo, ctx.keeper_repo, years, source)
        match result:
            case Ok(kc):
                ctx.conn.commit()
                player = ctx.player_repo.get_by_id(kc.player_id)
                name = f"{player.name_first} {player.name_last}" if player else "Unknown"
                typer.echo(f"Set keeper cost for {name}: ${cost:.0f} ({source}, {years}yr)")
            case Err(msg):
                typer.echo(f"Error: {msg}", err=True)
                raise typer.Exit(code=1)
