from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import console, print_error, print_run_detail, print_run_list
from fantasy_baseball_manager.cli.factory import build_runs_context
from fantasy_baseball_manager.models import RunManager

runs_app = typer.Typer(name="runs", help="Manage first-party model runs")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]
_OperationOpt = Annotated[str, typer.Option("--operation", help="Operation type (train, predict)")]


@runs_app.command("list")
def runs_list(
    model: Annotated[str | None, typer.Option("--model", help="Filter by model name")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """List recorded model runs."""
    with build_runs_context(data_dir) as ctx:
        records = ctx.repo.list(system=model)
    print_run_list(records)


@runs_app.command("show")
def runs_show(
    run: Annotated[str, typer.Argument(help="Run identifier (system/version)")],
    operation: _OperationOpt = "train",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show details of a model run."""
    parts = run.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid run format '{run}', expected 'system/version'")
        raise typer.Exit(code=1)
    system, version = parts

    with build_runs_context(data_dir) as ctx:
        record = ctx.repo.get(system, version, operation)
    if record is None:
        print_error(f"run '{run}' not found")
        raise typer.Exit(code=1)
    print_run_detail(record)


@runs_app.command("delete")
def runs_delete(
    run: Annotated[str, typer.Argument(help="Run identifier (system/version)")],
    operation: _OperationOpt = "train",
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Delete a model run and its artifacts."""
    parts = run.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid run format '{run}', expected 'system/version'")
        raise typer.Exit(code=1)
    system, version = parts

    with build_runs_context(data_dir) as ctx:
        record = ctx.repo.get(system, version, operation)
        if record is None:
            print_error(f"run '{run}' not found")
            raise typer.Exit(code=1)

        if not yes:
            typer.confirm(f"Delete run '{run}'?", abort=True)

        mgr = RunManager(model_run_repo=ctx.repo, artifacts_root=Path("."))
        mgr.delete_run(system, version, operation)
        ctx.conn.commit()
        console.print(f"[bold green]Deleted[/bold green] run '{run}'")
