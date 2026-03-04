from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import CohortBiasReport, ErrorDecompositionReport, FeatureGapReport


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


def print_feature_gap_report(report: FeatureGapReport) -> None:
    title = (
        f"Feature Gap Analysis — {report.system}/{report.version} "
        f"{report.target} ({report.player_type}s, {report.season})"
    )
    console.print(f"\n[bold]{title}[/bold]\n")

    in_model = [g for g in report.gaps if g.in_model]
    not_in_model = [g for g in report.gaps if not g.in_model]

    for label, gaps in [("In-Model Features", in_model), ("Not-In-Model Features", not_in_model)]:
        if not gaps:
            continue
        table = Table(title=label)
        table.add_column("Feature")
        table.add_column("KS Statistic", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Mean (Well)", justify="right")
        table.add_column("Mean (Poor)", justify="right")
        table.add_column("Sig?", justify="center")

        for gap in gaps:
            sig = "[bold red]*[/bold red]" if gap.p_value < 0.05 else ""
            table.add_row(
                gap.feature_name,
                f"{gap.ks_statistic:.3f}",
                f"{gap.p_value:.4f}",
                f"{gap.mean_well:.3f}",
                f"{gap.mean_poor:.3f}",
                sig,
            )

        console.print(table)


def print_cohort_bias_report(report: CohortBiasReport) -> None:
    title = (
        f"Cohort Bias — {report.dimension} — {report.system}/{report.version} "
        f"{report.target} ({report.player_type}s, {report.season})"
    )
    console.print(f"\n[bold]{title}[/bold]\n")

    table = Table(title=f"Bias by {report.dimension.title()}")
    table.add_column("Cohort")
    table.add_column("N", justify="right")
    table.add_column("Mean Residual", justify="right")
    table.add_column("Mean |Residual|", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("Significant", justify="center")

    for cohort in report.cohorts:
        residual_color = "red" if cohort.mean_residual > 0 else "green"
        style = "bold" if cohort.significant else ""
        sig_marker = "[bold red]*[/bold red]" if cohort.significant else ""
        table.add_row(
            cohort.cohort_label,
            str(cohort.n),
            f"[{residual_color}]{cohort.mean_residual:+.4f}[/{residual_color}]",
            f"{cohort.mean_abs_residual:.4f}",
            f"{cohort.rmse:.4f}",
            sig_marker,
            style=style,
        )

    console.print(table)


def print_cohort_bias_summary(reports: list[CohortBiasReport]) -> None:
    """Print a summary of the most biased significant cohorts across all dimensions."""
    for report in reports:
        print_cohort_bias_report(report)

    significant_cohorts: list[tuple[str, str, int, float, float]] = []
    for report in reports:
        for cohort in report.cohorts:
            if cohort.significant:
                significant_cohorts.append(
                    (report.dimension, cohort.cohort_label, cohort.n, cohort.mean_residual, cohort.rmse)
                )

    if not significant_cohorts:
        console.print("\n[dim]No statistically significant cohort biases found.[/dim]")
        return

    significant_cohorts.sort(key=lambda x: abs(x[3]), reverse=True)

    table = Table(title="Most Biased Cohorts (Significant)")
    table.add_column("Dimension")
    table.add_column("Cohort")
    table.add_column("N", justify="right")
    table.add_column("Mean Residual", justify="right")
    table.add_column("RMSE", justify="right")

    for dim, label, n, mean_r, rmse in significant_cohorts:
        residual_color = "red" if mean_r > 0 else "green"
        table.add_row(
            dim,
            label,
            str(n),
            f"[{residual_color}]{mean_r:+.4f}[/{residual_color}]",
            f"{rmse:.4f}",
        )

    console.print(table)
