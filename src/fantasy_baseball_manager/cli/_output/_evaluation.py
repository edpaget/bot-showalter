from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import (
    _build_two_system_row,
    console,
)
from fantasy_baseball_manager.domain import summarize_comparison

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        ComparisonResult,
        RegressionCheckResult,
        StratifiedComparisonResult,
        SystemMetrics,
    )
    from fantasy_baseball_manager.services import GateResult


def print_system_metrics(metrics: SystemMetrics) -> None:
    """Print evaluation results in tabular format."""
    console.print(f"Evaluation: [bold]{metrics.system}[/bold] v{metrics.version} [dim]({metrics.source_type})[/dim]")
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Stat")
    table.add_column("RMSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("r", justify="right")
    table.add_column("\u03c1", justify="right")
    table.add_column("R\u00b2", justify="right")
    table.add_column("Bias", justify="right")
    table.add_column("N", justify="right")
    for stat_name in sorted(metrics.metrics):
        m = metrics.metrics[stat_name]
        table.add_row(
            stat_name,
            f"{m.rmse:.4f}",
            f"{m.mae:.4f}",
            f"{m.correlation:.4f}",
            f"{m.rank_correlation:.4f}",
            f"{m.r_squared:.4f}",
            f"{m.mean_error:+.4f}",
            str(m.n),
        )
    console.print(table)


def _print_tail_accuracy_section(result: ComparisonResult) -> None:
    """Print tail accuracy table when systems have tail data."""
    systems_with_tail = [s for s in result.systems if s.tail is not None]
    if not systems_with_tail:
        return

    console.print()
    console.print("[bold]Tail accuracy (top-N RMSE)[/bold]")

    # Collect all stats and ns
    all_stats: set[str] = set()
    ns: tuple[int, ...] = ()
    for s in systems_with_tail:
        assert s.tail is not None  # noqa: S101 — type narrowing
        ns = s.tail.ns
        all_stats.update(s.tail.rmse_by_stat.keys())

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Stat")
    for s in systems_with_tail:
        label = f"{s.system}/{s.version}"
        for n in sorted(ns):
            table.add_column(f"{label} top-{n}", justify="right")

    for stat_name in sorted(all_stats):
        values: list[str] = []
        for s in systems_with_tail:
            assert s.tail is not None  # noqa: S101
            stat_rmse = s.tail.rmse_by_stat.get(stat_name, {})
            for n in sorted(ns):
                rmse = stat_rmse.get(n)
                values.append(f"{rmse:.4f}" if rmse is not None else "\u2014")
        table.add_row(stat_name, *values)

    console.print(table)


def print_comparison_result(result: ComparisonResult) -> None:
    """Print comparison table across systems."""
    console.print(f"Comparison — season [bold]{result.season}[/bold]")

    if len(result.systems) == 2:
        summary = summarize_comparison(result)
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Stat")
        table.add_column(f"{summary.baseline_label} RMSE", justify="right")
        table.add_column(f"{summary.candidate_label} RMSE", justify="right")
        table.add_column("\u0394", justify="right")
        table.add_column("%\u0394", justify="right")
        table.add_column(f"{summary.baseline_label} R\u00b2", justify="right")
        table.add_column(f"{summary.candidate_label} R\u00b2", justify="right")
        table.add_column("\u0394", justify="right")
        table.add_column("%\u0394", justify="right")
        table.add_column(f"{summary.baseline_label} \u03c1", justify="right")
        table.add_column(f"{summary.candidate_label} \u03c1", justify="right")
        table.add_column("\u0394", justify="right")
        table.add_column("%\u0394", justify="right")
        for rec in summary.records:
            table.add_row(*_build_two_system_row(rec))
        console.print(table)
        total = len(summary.records)
        console.print(
            f"[bold]{summary.candidate_label} vs {summary.baseline_label}:[/bold]"
            f" wins {summary.rmse_wins}/{total} stats on RMSE,"
            f" {summary.r_squared_wins}/{total} on R\u00b2,"
            f" {summary.rank_correlation_wins}/{total} on \u03c1"
        )
        _print_tail_accuracy_section(result)
        return

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Stat")
    for sys_metrics in result.systems:
        label = f"{sys_metrics.system}/{sys_metrics.version}"
        table.add_column(f"{label} RMSE", justify="right")
        table.add_column(f"{label} R\u00b2", justify="right")
        table.add_column(f"{label} \u03c1", justify="right")
        table.add_column(f"{label} Bias", justify="right")
    for stat_name in result.stats:
        values: list[str] = []
        for sys_metrics in result.systems:
            m = sys_metrics.metrics.get(stat_name)
            values.append(f"{m.rmse:.4f}" if m else "\u2014")
            values.append(f"{m.r_squared:.4f}" if m else "\u2014")
            values.append(f"{m.rank_correlation:.4f}" if m else "\u2014")
            values.append(f"{m.mean_error:+.4f}" if m else "\u2014")
        table.add_row(stat_name, *values)
    console.print(table)
    _print_tail_accuracy_section(result)


def print_stratified_comparison_result(result: StratifiedComparisonResult) -> None:
    """Print comparison tables per cohort."""
    console.print(
        f"Stratified comparison — season [bold]{result.season}[/bold], dimension [bold]{result.dimension}[/bold]"
    )
    console.print()
    for label, comp in result.cohorts.items():
        console.print(f"  [bold]{label}[/bold]")
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Stat")
        for sys_metrics in comp.systems:
            label = f"{sys_metrics.system}/{sys_metrics.version}"
            table.add_column(f"{label} MAE", justify="right")
            table.add_column(f"{label} R²", justify="right")
        for stat_name in comp.stats:
            values: list[str] = []
            for sys_metrics in comp.systems:
                m = sys_metrics.metrics.get(stat_name)
                values.append(f"{m.mae:.4f}" if m else "—")
                values.append(f"{m.r_squared:.4f}" if m else "—")
            table.add_row(stat_name, *values)
        console.print(table)
        console.print()


def print_regression_check_result(check: RegressionCheckResult) -> None:
    """Print a pass/fail regression check verdict."""
    console.print()
    if check.passed:
        console.print(f"[bold green]PASS[/bold green]: {check.explanation}")
    else:
        console.print(f"[bold red]FAIL[/bold red]: {check.explanation}")


def print_gate_result(result: GateResult) -> None:
    """Print a regression gate summary table."""
    console.print()
    console.print(f"[bold]Regression Gate — {result.model_name}[/bold]")
    console.print(f"Baseline: {result.baseline}")
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
