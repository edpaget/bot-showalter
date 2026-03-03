from typing import TYPE_CHECKING

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import LoadLog


def print_import_result(log: LoadLog) -> None:
    console.print(f"[bold green]Import complete:[/bold green] {log.rows_loaded} projections loaded")
    console.print(f"  Source: {log.source_detail}")
    console.print(f"  Status: {log.status}")


def print_ingest_result(log: LoadLog) -> None:
    console.print(f"[bold green]Ingest complete:[/bold green] {log.rows_loaded} rows loaded into {log.target_table}")
    console.print(f"  Source: {log.source_detail}")
    console.print(f"  Status: {log.status}")
    if log.error_message:
        console.print(f"  [red]Error: {log.error_message}[/red]")
