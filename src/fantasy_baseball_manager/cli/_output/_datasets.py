from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.services import DatasetInfo


def print_dataset_list(datasets: list[DatasetInfo]) -> None:
    """Print a table of cached datasets."""
    if not datasets:
        console.print("No cached datasets found.")
        return
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Feature Set")
    table.add_column("Version")
    table.add_column("Split")
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    table.add_column("Seasons")
    table.add_column("Created")
    for d in datasets:
        seasons_str = ", ".join(str(s) for s in d.seasons) if d.seasons else ""
        table.add_row(
            d.feature_set_name,
            d.feature_set_version,
            d.split or "—",
            d.table_name,
            str(d.row_count),
            seasons_str,
            d.created_at,
        )
    console.print(table)
