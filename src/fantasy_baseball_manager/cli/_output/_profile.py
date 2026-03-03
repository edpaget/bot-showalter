from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        ColumnProfile,
        CorrelationScanResult,
        MultiColumnRanking,
        StabilityResult,
        TargetStability,
    )


def print_column_profiles(profiles: list[ColumnProfile]) -> None:
    """Print column profile results as a Rich table."""
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Column", justify="left")
    table.add_column("Season", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Null%", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("P10", justify="right")
    table.add_column("P25", justify="right")
    table.add_column("P75", justify="right")
    table.add_column("P90", justify="right")
    table.add_column("Skew", justify="right")

    for p in profiles:
        table.add_row(
            p.column,
            str(p.season),
            str(p.count),
            f"{p.null_pct:.1f}",
            f"{p.mean:.3f}",
            f"{p.median:.3f}",
            f"{p.std:.3f}",
            f"{p.min:.3f}",
            f"{p.max:.3f}",
            f"{p.p10:.3f}",
            f"{p.p25:.3f}",
            f"{p.p75:.3f}",
            f"{p.p90:.3f}",
            f"{p.skewness:.3f}",
        )

    console.print(table)


def print_column_ranking(rankings: list[MultiColumnRanking]) -> None:
    """Print column ranking summary table."""
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Column", justify="left")
    table.add_column("Avg |Pearson|", justify="right")
    table.add_column("Avg |Spearman|", justify="right")

    for r in rankings:
        table.add_row(
            r.column_spec,
            f"{r.avg_abs_pearson:.3f}",
            f"{r.avg_abs_spearman:.3f}",
        )

    console.print(table)


def print_correlation_results(result: CorrelationScanResult) -> None:
    """Print correlation scan results as Rich tables."""
    for season_result in result.per_season:
        console.print(f"\n[bold]Season {season_result.season}[/bold]")
        _print_correlation_table(season_result.correlations)

    console.print("\n[bold]Pooled (all seasons)[/bold]")
    _print_correlation_table(result.pooled.correlations)


def _print_correlation_table(correlations: tuple) -> None:
    """Print a single correlation table."""
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Target", justify="left")
    table.add_column("Pearson r", justify="right")
    table.add_column("Spearman rho", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("n", justify="right")

    for c in correlations:
        table.add_row(
            c.target,
            f"{c.pearson_r:.3f}",
            f"{c.spearman_rho:.3f}",
            f"{c.pearson_p:.4f}",
            str(c.n),
        )

    console.print(table)


def _classify_color(classification: str) -> str:
    """Return a Rich color tag for a stability classification."""
    if classification == "stable":
        return "green"
    if classification == "unstable":
        return "red"
    return "yellow"


def print_stability_result(result: StabilityResult) -> None:
    """Print stability results — single-target detail or multi-target matrix."""
    if len(result.target_stabilities) == 1:
        _print_single_target_stability(result.target_stabilities[0])
    else:
        _print_stability_matrix(result)


def _print_single_target_stability(ts: TargetStability) -> None:
    """Print per-season detail and summary for a single target."""
    console.print(f"\n[bold]Stability: {ts.target}[/bold]")

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Season", justify="right")
    table.add_column("Pearson r", justify="right")
    for season, r in ts.per_season_r:
        table.add_row(str(season), f"{r:.3f}")
    console.print(table)

    color = _classify_color(ts.classification)
    cv_display = f"{ts.cv:.3f}" if ts.cv >= 0 else "n/a (near-zero mean)"
    console.print(f"  Mean r:  {ts.mean_r:.3f}")
    console.print(f"  Std:     {ts.std_r:.3f}")
    console.print(f"  CV:      {cv_display}")
    console.print(f"  Rating:  [{color}]{ts.classification}[/{color}]")


def _print_stability_matrix(result: StabilityResult) -> None:
    """Print a matrix table of stability across all targets."""
    console.print(f"\n[bold]Temporal Stability: {result.column_spec} ({result.player_type})[/bold]")

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Target", justify="left")
    table.add_column("Mean r", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("CV", justify="right")
    table.add_column("Stability", justify="center")

    for season in result.seasons:
        table.add_column(str(season), justify="right")

    for ts in result.target_stabilities:
        color = _classify_color(ts.classification)
        cv_display = f"{ts.cv:.3f}" if ts.cv >= 0 else "n/a"
        row: list[str] = [
            ts.target,
            f"{ts.mean_r:.3f}",
            f"{ts.std_r:.3f}",
            cv_display,
            f"[{color}]{ts.classification}[/{color}]",
        ]
        season_map = dict(ts.per_season_r)
        for season in result.seasons:
            r = season_map.get(season)
            row.append(f"{r:.3f}" if r is not None else "-")
        table.add_row(*row)

    console.print(table)
