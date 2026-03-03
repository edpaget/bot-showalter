from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import CandidateValue


def print_candidate_values(values: list[CandidateValue]) -> None:
    """Print candidate aggregation values as a Rich table."""
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Player ID", justify="right")
    table.add_column("Season", justify="right")
    table.add_column("Value", justify="right")

    null_count = 0
    for v in values:
        if v.value is None:
            null_count += 1
            value_str = "NULL"
        else:
            value_str = f"{v.value:.4f}"
        table.add_row(str(v.player_id), str(v.season), value_str)

    console.print(table)
    console.print(f"\n{len(values)} player-seasons ({null_count} with NULL values)")
