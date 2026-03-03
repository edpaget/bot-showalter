"""Output formatting for validation gate commands."""

from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.services import PreflightResult


def print_preflight_result(result: PreflightResult) -> None:
    """Print pre-flight confidence check results."""
    confidence_colors = {"high": "green", "medium": "yellow", "low": "red"}
    color = confidence_colors.get(result.confidence, "")

    console.print("[bold]Pre-flight Check[/bold]")
    console.print()

    if result.details:
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Target")
        table.add_column("Win Rate", justify="right")
        table.add_column("Mean Delta", justify="right")
        table.add_column("Delta Std", justify="right")

        for detail in result.details:
            wr_color = "green" if detail.win_rate >= 0.75 else "yellow" if detail.win_rate >= 0.60 else "red"
            delta_color = "green" if detail.mean_delta < 0 else "red"
            table.add_row(
                detail.target,
                f"[{wr_color}]{detail.win_rate:.0%}[/{wr_color}]",
                f"[{delta_color}]{detail.mean_delta:+.6f}[/{delta_color}]",
                f"{detail.delta_std:.6f}",
            )

        console.print(table)
        console.print()

    console.print(f"Confidence: [{color}]{result.confidence}[/{color}]")
    console.print(f"Recommendation: [{color}]{result.recommendation}[/{color}]")
