from collections import defaultdict
from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import BinnedValue, BinTargetMean, CandidateValue, MultiColumnRanking


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


def print_interaction_scan_results(results: list[tuple[str, MultiColumnRanking]]) -> None:
    """Print ranked interaction scan results as a Rich table."""
    table = Table(title="Interaction Scan Results", show_edge=False, pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Operation")
    table.add_column("Avg |Pearson|", justify="right")
    table.add_column("Avg |Spearman|", justify="right")

    for i, (op, ranking) in enumerate(results, 1):
        table.add_row(
            str(i),
            op,
            f"{ranking.avg_abs_pearson:.4f}",
            f"{ranking.avg_abs_spearman:.4f}",
        )

    console.print(table)


def print_binned_summary(values: list[BinnedValue]) -> None:
    """Print binned value summary as a Rich table: Bin | Count | Min | Max."""
    by_bin: dict[str, list[float]] = defaultdict(list)
    for v in values:
        by_bin[v.bin_label].append(v.value)

    table = Table(title="Binned Summary", show_edge=False, pad_edge=False)
    table.add_column("Bin")
    table.add_column("Count", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for label in sorted(by_bin):
        vals = by_bin[label]
        table.add_row(label, str(len(vals)), f"{min(vals):.4f}", f"{max(vals):.4f}")

    console.print(table)
    console.print(f"\n{len(values)} player-seasons across {len(by_bin)} bins")


def print_bin_target_means(means: list[BinTargetMean]) -> None:
    """Print bin target means as a pivoted Rich table: rows = bins, columns = targets."""
    if not means:
        console.print("No target means to display")
        return

    # Collect unique bins and targets
    bins = sorted({m.bin_label for m in means})
    targets = sorted({m.target for m in means})

    # Build lookup
    lookup: dict[tuple[str, str], BinTargetMean] = {(m.bin_label, m.target): m for m in means}

    table = Table(title="Within-Bin Target Means", show_edge=False, pad_edge=False)
    table.add_column("Bin")
    for t in targets:
        table.add_column(t, justify="right")

    for b in bins:
        row: list[str] = [b]
        for t in targets:
            entry = lookup.get((b, t))
            if entry is not None:
                row.append(f"{entry.mean:.4f} (n={entry.count})")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)
