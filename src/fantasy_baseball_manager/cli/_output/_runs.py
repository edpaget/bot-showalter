import json
from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import ModelRunRecord


def print_run_list(records: list[ModelRunRecord]) -> None:
    """Print a table of model runs."""
    if not records:
        console.print("No runs found.")
        return
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("System")
    table.add_column("Version")
    table.add_column("Operation")
    table.add_column("Created")
    table.add_column("Tags")
    for r in records:
        tags_str = ", ".join(f"{k}={v}" for k, v in r.tags_json.items()) if r.tags_json else ""
        table.add_row(r.system, r.version, r.operation, r.created_at, tags_str)
    console.print(table)


def print_run_detail(record: ModelRunRecord) -> None:
    """Print full details of a model run."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("System", record.system)
    table.add_row("Version", record.version)
    table.add_row("Operation", record.operation)
    table.add_row("Created", record.created_at)
    table.add_row("Git Commit", record.git_commit or "N/A")
    table.add_row("Artifact Type", record.artifact_type)
    table.add_row("Artifact Path", record.artifact_path or "N/A")
    if record.config_json:
        table.add_row("Config", json.dumps(record.config_json, indent=2))
    if record.metrics_json:
        table.add_row("Metrics", json.dumps(record.metrics_json, indent=2))
    if record.tags_json:
        tags_str = ", ".join(f"{k}={v}" for k, v in record.tags_json.items())
        table.add_row("Tags", tags_str)
    console.print(table)
