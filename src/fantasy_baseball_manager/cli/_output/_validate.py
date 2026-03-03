"""Output formatting for validation gate commands."""

from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.services import PreflightResult, ValidationResult


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


def print_validation_result(result: ValidationResult) -> None:
    """Print full validation orchestrator results."""
    console.print()
    console.print(f"[bold]Validation Gate — {result.model_name}[/bold]")
    console.print(f"{result.old_version} → {result.new_version}")
    console.print()

    if result.preflight is not None and result.preflight.confidence == "low":
        console.print("[yellow]⚠ Pre-flight confidence is LOW — proceeding anyway[/yellow]")
        console.print()

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Season", justify="right")
    table.add_column("Segment")
    table.add_column("RMSE")
    table.add_column("ρ")
    table.add_column("Verdict")

    for seg in result.segments:
        rmse_label = "[green]PASS[/green]" if seg.check.rmse_passed else "[red]FAIL[/red]"
        rho_label = "[green]PASS[/green]" if seg.check.rank_correlation_passed else "[red]FAIL[/red]"
        verdict_label = "[bold green]PASS[/bold green]" if seg.check.passed else "[bold red]FAIL[/bold red]"
        table.add_row(str(seg.season), seg.segment, rmse_label, rho_label, verdict_label)

    console.print(table)
    console.print()

    failed_count = sum(1 for s in result.segments if not s.check.passed)
    if result.passed:
        console.print("[bold green]OVERALL: PASS[/bold green]")
    else:
        console.print(f"[bold red]OVERALL: FAIL[/bold red] — {failed_count} check(s) failed")
