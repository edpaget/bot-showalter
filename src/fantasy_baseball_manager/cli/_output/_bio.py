from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import PlayerSummary


def print_player_summaries(summaries: list[PlayerSummary]) -> None:
    """Print player biography results as a Rich table."""
    if not summaries:
        console.print("No players found.")
        return
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Name")
    table.add_column("Team")
    table.add_column("Pos")
    table.add_column("Age", justify="right")
    table.add_column("B/T")
    table.add_column("Exp", justify="right")
    for s in summaries:
        age_str = str(s.age) if s.age is not None else ""
        bt = f"{s.bats or '?'}/{s.throws or '?'}"
        table.add_row(s.name, s.team, s.primary_position, age_str, bt, str(s.experience))
    console.print(table)
