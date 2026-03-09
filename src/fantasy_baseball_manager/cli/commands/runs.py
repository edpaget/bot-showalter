from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._defaults import _DataDirOpt  # noqa: TC001 — used at runtime by typer
from fantasy_baseball_manager.cli._output import (
    console,
    print_error,
    print_run_detail,
    print_run_diff,
    print_run_inspect,
    print_run_list,
)
from fantasy_baseball_manager.cli.factory import build_runs_context
from fantasy_baseball_manager.models import RunManager

runs_app = typer.Typer(name="runs", help="Manage first-party model runs")

_OperationOpt = Annotated[str, typer.Option("--operation", help="Operation type (train, predict)")]


@runs_app.command("list")
def runs_list(  # pragma: no cover
    model: Annotated[str | None, typer.Option("--model", help="Filter by model name")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """List recorded model runs."""
    with build_runs_context(data_dir) as ctx:
        records = ctx.repo.list(system=model)
    print_run_list(records)


@runs_app.command("show")
def runs_show(  # pragma: no cover
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


@runs_app.command("inspect")
def runs_inspect(  # pragma: no cover
    system: Annotated[str, typer.Argument(help="Model system name")],
    version: Annotated[str | None, typer.Option("--version", help="Run version (default: latest)")] = None,
    section: Annotated[
        str | None, typer.Option("--section", help="Show only this section (config, metrics, tags)")
    ] = None,
    operation: _OperationOpt = "train",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Inspect a model run with structured output."""
    with build_runs_context(data_dir) as ctx:
        if version is not None:
            record = ctx.repo.get(system, version, operation)
        else:
            record = ctx.repo.get_latest(system, operation)
    if record is None:
        label = f"{system}/{version}" if version else f"{system} (latest)"
        print_error(f"run '{label}' not found")
        raise typer.Exit(code=1)
    print_run_inspect(record, section=section)


@runs_app.command("diff")
def runs_diff(  # pragma: no cover
    run_a: Annotated[str, typer.Argument(help="First run (system/version or system for latest)")],
    run_b: Annotated[str, typer.Argument(help="Second run (system/version or system for latest)")],
    operation: _OperationOpt = "train",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Compare two model runs side by side."""
    with build_runs_context(data_dir) as ctx:
        record_a = _resolve_run(ctx.repo, run_a, operation)
        record_b = _resolve_run(ctx.repo, run_b, operation)
    if record_a is None:
        print_error(f"run '{run_a}' not found")
        raise typer.Exit(code=1)
    if record_b is None:
        print_error(f"run '{run_b}' not found")
        raise typer.Exit(code=1)
    print_run_diff(record_a, record_b)


def _resolve_run(repo, run_spec: str, operation: str):  # pragma: no cover
    """Resolve a run spec like 'system/version' or 'system' (latest)."""
    if "/" in run_spec:
        system, version = run_spec.split("/", 1)
        return repo.get(system, version, operation)
    return repo.get_latest(run_spec, operation)


@runs_app.command("delete")
def runs_delete(  # pragma: no cover
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
