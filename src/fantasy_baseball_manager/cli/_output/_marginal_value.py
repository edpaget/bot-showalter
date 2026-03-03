from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.services import MarginalValueResult


def print_marginal_value_results(results: list[MarginalValueResult]) -> None:
    for result in results:
        console.print(f"\n[bold]Candidate: {result.candidate}[/bold]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Target")
        table.add_column("Baseline RMSE", justify="right")
        table.add_column("With Candidate", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Delta %", justify="right")

        for d in result.deltas:
            color = "green" if d.delta < 0 else "red" if d.delta > 0 else ""
            delta_str = f"[{color}]{d.delta:+.4f}[/{color}]" if color else f"{d.delta:+.4f}"
            pct_str = f"[{color}]{d.delta_pct:+.1f}%[/{color}]" if color else f"{d.delta_pct:+.1f}%"
            table.add_row(
                d.target,
                f"{d.baseline_rmse:.4f}",
                f"{d.candidate_rmse:.4f}",
                delta_str,
                pct_str,
            )

        console.print(table)
        verdict_color = "green" if result.n_improved > 0 else "yellow"
        console.print(
            f"  [{verdict_color}]Improves {result.n_improved}/{result.n_total} targets[/{verdict_color}]"
            f"  (avg delta: {result.avg_delta_pct:+.1f}%)"
        )

    if len(results) > 1:
        console.print("\n[bold]Candidate Ranking[/bold] (by avg delta %)")
        rank_table = Table(show_header=True, header_style="bold")
        rank_table.add_column("Rank", justify="right")
        rank_table.add_column("Candidate")
        rank_table.add_column("Avg Delta %", justify="right")
        rank_table.add_column("Improved", justify="right")

        for i, r in enumerate(results, 1):
            color = "green" if r.avg_delta_pct < 0 else "red" if r.avg_delta_pct > 0 else ""
            pct_str = f"[{color}]{r.avg_delta_pct:+.1f}%[/{color}]" if color else f"{r.avg_delta_pct:+.1f}%"
            rank_table.add_row(str(i), r.candidate, pct_str, f"{r.n_improved}/{r.n_total}")

        console.print(rank_table)
