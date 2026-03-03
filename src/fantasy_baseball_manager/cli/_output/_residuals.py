from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import ErrorDecompositionReport


def print_error_decomposition_report(report: ErrorDecompositionReport) -> None:
    title = f"Worst Misses — {report.system}/{report.version} {report.target} ({report.player_type}s, {report.season})"
    console.print(f"\n[bold]{title}[/bold]\n")

    volume_label = "IP" if report.player_type == "pitcher" else "PA"

    table = Table(title="Top Misses")
    table.add_column("Rank", justify="right")
    table.add_column("Player")
    table.add_column("Predicted", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Residual", justify="right")
    table.add_column("Age", justify="right")
    table.add_column(volume_label, justify="right")

    for i, miss in enumerate(report.top_misses, 1):
        age_str = f"{miss.feature_values['age']:.0f}" if "age" in miss.feature_values else "-"
        vol_key = "ip" if report.player_type == "pitcher" else "pa"
        vol_str = f"{miss.feature_values[vol_key]:.0f}" if vol_key in miss.feature_values else "-"
        residual_color = "red" if miss.residual > 0 else "green"
        table.add_row(
            str(i),
            miss.player_name,
            f"{miss.predicted:.3f}",
            f"{miss.actual:.3f}",
            f"[{residual_color}]{miss.residual:+.3f}[/{residual_color}]",
            age_str,
            vol_str,
        )

    console.print(table)

    summary = report.summary
    summary_table = Table(title="Miss Population Summary")
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")

    if summary.mean_age is not None:
        summary_table.add_row("Mean Age", f"{summary.mean_age:.1f}")
    summary_table.add_row(f"Mean {volume_label}", f"{summary.mean_volume:.0f}")

    if summary.position_distribution:
        pos_str = ", ".join(f"{pos}: {count}" for pos, count in sorted(summary.position_distribution.items()))
        summary_table.add_row("Positions", pos_str)

    console.print(summary_table)

    if summary.distinguishing_features:
        feat_table = Table(title="Distinguishing Features")
        feat_table.add_column("Feature")
        feat_table.add_column("Mean (Miss Group)", justify="right")
        feat_table.add_column("Mean (Rest)", justify="right")
        feat_table.add_column("Difference", justify="right")

        for feat in summary.distinguishing_features[:10]:
            feat_table.add_row(
                feat.feature_name,
                f"{feat.mean_miss_group:.3f}",
                f"{feat.mean_rest:.3f}",
                f"{feat.difference:.3f}",
            )

        console.print(feat_table)
