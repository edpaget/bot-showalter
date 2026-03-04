from typing import Annotated

import typer

from fantasy_baseball_manager.cli._dispatcher import dispatch
from fantasy_baseball_manager.cli._output import console, print_dataset_list, print_error, print_prepare_result
from fantasy_baseball_manager.cli.factory import build_datasets_context, build_model_context
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.domain import Err, Ok
from fantasy_baseball_manager.models import PrepareResult

datasets_app = typer.Typer(name="datasets", help="Manage cached feature-set datasets")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]
_ModelArg = Annotated[str, typer.Argument(help="Name of the projection model")]
_SeasonOpt = Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")]


@datasets_app.command("list")
def datasets_list(  # pragma: no cover
    name: Annotated[str | None, typer.Option("--name", help="Filter by feature set name")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show all materialized feature sets and their datasets."""
    with build_datasets_context(data_dir) as ctx:
        datasets = ctx.catalog.list_by_feature_set_name(name) if name else ctx.catalog.list_all()
    print_dataset_list(datasets)


@datasets_app.command("drop")
def datasets_drop(  # pragma: no cover
    name: Annotated[str | None, typer.Option("--name", help="Feature set name to drop")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Drop all cached datasets")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Drop cached datasets."""
    if not name and not all_:
        print_error("provide --name or --all")
        raise typer.Exit(code=1)

    with build_datasets_context(data_dir) as ctx:
        if not yes:
            target = f"feature set '{name}'" if name else "ALL cached datasets"
            typer.confirm(f"Drop {target}?", abort=True)

        if all_:
            count = ctx.catalog.drop_all()
        else:
            assert name is not None  # noqa: S101 - type narrowing
            count = ctx.catalog.drop_by_feature_set_name(name)

        ctx.conn.commit()

    if count == 0:
        console.print("No datasets found to drop.")
    else:
        console.print(f"[bold green]Dropped[/bold green] {count} dataset(s)")


@datasets_app.command("rebuild")
def datasets_rebuild(  # pragma: no cover
    model: _ModelArg,
    season: _SeasonOpt = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Drop a model's cached datasets, then re-materialize via prepare."""
    prefix = model.replace("-", "_") + "_"

    with build_datasets_context(data_dir) as ctx:
        datasets = ctx.catalog.list_all()
        matching = [d for d in datasets if d.feature_set_name.startswith(prefix)]

        if not matching:
            console.print(f"No cached datasets found for model '{model}'.")
        else:
            if not yes:
                typer.confirm(f"Drop {len(matching)} dataset(s) for model '{model}'?", abort=True)
            count = ctx.catalog.drop_by_name_prefix(prefix)
            ctx.conn.commit()
            console.print(f"[bold green]Dropped[/bold green] {count} dataset(s)")

    config = load_config(model_name=model, seasons=season)
    with build_model_context(model, config) as ctx:
        match dispatch("prepare", ctx.model, config):
            case Ok(PrepareResult() as r):
                print_prepare_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)
