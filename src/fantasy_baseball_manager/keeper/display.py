"""Display functions for keeper CLI output.

This module contains Rich table formatting for keeper rankings and recommendations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from fantasy_baseball_manager.keeper.models import KeeperSurplus

console = Console()


def display_table(ranked: list[KeeperSurplus], year: int, title: str) -> None:
    """Display a keeper ranking table with a title."""
    console.print(f"\n[bold]{title} ({year}):[/bold]\n")
    print_keeper_table(ranked)


def print_keeper_table(rows: list[KeeperSurplus]) -> None:
    """Print a formatted table of keeper scores."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Rk", justify="right")
    table.add_column("Name")
    table.add_column("Pos")
    table.add_column("Value", justify="right")
    table.add_column("Repl", justify="right")
    table.add_column("Surplus", justify="right")
    table.add_column("Slot", justify="right")

    for i, ks in enumerate(rows, start=1):
        pos_str = "/".join(ks.eligible_positions) if ks.eligible_positions else "-"
        table.add_row(
            str(i),
            ks.name,
            pos_str,
            f"{ks.player_value:.1f}",
            f"{ks.replacement_value:.1f}",
            f"{ks.surplus_value:.1f}",
            str(ks.assigned_slot),
        )

    console.print(table)
