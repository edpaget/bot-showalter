from collections import defaultdict
from typing import TYPE_CHECKING

from rich.console import Console  # noqa: TC002 — used at runtime in default param
from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console
from fantasy_baseball_manager.domain import (
    ClassifiedPlayer,
    ConfidenceReport,
    PlayerConfidence,
    VarianceClassification,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        ADPAccuracyReport,
        ADPMoversReport,
        PlayerStatDelta,
        Projection,
        ResidualAnalysisReport,
        ResidualPersistenceReport,
        TrueTalentQualityReport,
        ValueOverADPReport,
    )


def print_performance_report(title: str, deltas: list[PlayerStatDelta]) -> None:
    """Print a performance report table."""
    if not deltas:
        console.print("No results found.")
        return
    console.print(f"[bold]{title}[/bold]")
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Player")
    table.add_column("Stat")
    table.add_column("Actual", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Pctile", justify="right")
    for d in deltas:
        color = "green" if d.delta > 0 else "red" if d.delta < 0 else ""
        delta_str = f"[{color}]{d.delta:+.3f}[/{color}]" if color else f"{d.delta:+.3f}"
        table.add_row(
            d.player_name,
            d.stat_name,
            f"{d.actual:.3f}",
            f"{d.expected:.3f}",
            delta_str,
            f"{d.percentile:.0f}",
        )
    console.print(table)


def print_talent_delta_report(
    title: str,
    deltas: list[PlayerStatDelta],
    *,
    top: int | None = None,
    console: Console = console,
) -> None:
    """Print a talent-delta report grouped by stat with regression/buy-low sections."""
    if not deltas:
        console.print("No results found.")
        return

    console.print(f"[bold]{title}[/bold]")
    console.print()

    by_stat: dict[str, list[PlayerStatDelta]] = defaultdict(list)
    for d in deltas:
        by_stat[d.stat_name].append(d)

    for stat_name, stat_deltas in by_stat.items():
        regression = [d for d in stat_deltas if d.performance_delta > 0]
        buylow = [d for d in stat_deltas if d.performance_delta < 0]

        regression.sort(key=lambda d: d.performance_delta, reverse=True)
        buylow.sort(key=lambda d: d.performance_delta)

        if top is not None:
            regression = regression[:top]
            buylow = buylow[:top]

        if regression:
            console.print(f"[bold]{stat_name}[/bold] — Regression Candidates (actual > true talent)")
            table = Table(show_edge=False, pad_edge=False)
            table.add_column("Player")
            table.add_column("Actual", justify="right")
            table.add_column("Talent", justify="right")
            table.add_column("Delta", justify="right")
            table.add_column("Pctile", justify="right")
            for d in regression:
                table.add_row(
                    d.player_name,
                    f"{d.actual:.3f}",
                    f"{d.expected:.3f}",
                    f"[red]{d.delta:+.3f}[/red]" if d.delta < 0 else f"[green]{d.delta:+.3f}[/green]",
                    f"{d.percentile:.0f}",
                )
            console.print(table)
            console.print()

        if buylow:
            console.print(f"[bold]{stat_name}[/bold] — Buy-Low Targets (actual < true talent)")
            table = Table(show_edge=False, pad_edge=False)
            table.add_column("Player")
            table.add_column("Actual", justify="right")
            table.add_column("Talent", justify="right")
            table.add_column("Delta", justify="right")
            table.add_column("Pctile", justify="right")
            for d in buylow:
                table.add_row(
                    d.player_name,
                    f"{d.actual:.3f}",
                    f"{d.expected:.3f}",
                    f"[red]{d.delta:+.3f}[/red]" if d.delta < 0 else f"[green]{d.delta:+.3f}[/green]",
                    f"{d.percentile:.0f}",
                )
            console.print(table)
            console.print()


def print_talent_quality_report(reports: list[TrueTalentQualityReport]) -> None:
    """Print true-talent quality evaluation reports."""
    if not reports:
        console.print("No results found.")
        return

    for report in reports:
        console.print(
            f"[bold]True-Talent Quality[/bold] — {report.system}/{report.version}"
            f" ({report.season_n} -> {report.season_n1}, {report.player_type}s)"
        )
        console.print()

        s = report.summary

        def _pass_label(passes: int, total: int) -> str:
            ratio = passes / total if total > 0 else 0
            color = "green" if ratio >= 0.5 else "red"
            return f"[{color}]{passes}/{total}[/{color}]"

        console.print("[bold]Summary:[/bold]")
        console.print(
            f"  Predictive validity:      {_pass_label(s.predictive_validity_passes, s.predictive_validity_total)}"
            " pass (model > raw)"
        )
        res_label = _pass_label(s.residual_non_persistence_passes, s.residual_non_persistence_total)
        console.print(f"  Residual non-persistence: {res_label} pass (|corr| < 0.15)")
        console.print(
            f"  Shrinkage quality:        {_pass_label(s.shrinkage_passes, s.shrinkage_total)} pass (ratio < 0.9)"
        )
        console.print(
            f"  R-squared:                {_pass_label(s.r_squared_passes, s.r_squared_total)} pass (R² > 0.7)"
        )
        console.print(
            f"  Regression rate:          {_pass_label(s.regression_rate_passes, s.regression_rate_total)}"
            " pass (rate > 0.80)"
        )
        console.print()

        if report.stat_metrics:
            table = Table(show_edge=False, pad_edge=False)
            table.add_column("Stat")
            table.add_column("Model->N+1", justify="right")
            table.add_column("Raw->N+1", justify="right")
            table.add_column("Res.Corr", justify="right")
            table.add_column("Shrink", justify="right")
            table.add_column("R²", justify="right")
            table.add_column("Regr.Rate", justify="right")
            table.add_column("N", justify="right")
            table.add_column("Ret", justify="right")
            for m in report.stat_metrics:
                model_color = "green" if m.predictive_validity_pass else "red"
                res_color = "green" if m.residual_non_persistence_pass else "red"
                shrink_color = "green" if m.shrinkage_pass else "red"
                r2_color = "green" if m.r_squared_pass else "red"
                regr_color = "green" if m.regression_rate_pass else "red"
                table.add_row(
                    m.stat_name,
                    f"[{model_color}]{m.model_next_season_corr:.3f}[/{model_color}]",
                    f"{m.raw_next_season_corr:.3f}",
                    f"[{res_color}]{m.residual_yoy_corr:.3f}[/{res_color}]",
                    f"[{shrink_color}]{m.shrinkage_ratio:.2f}[/{shrink_color}]",
                    f"[{r2_color}]{m.r_squared:.3f}[/{r2_color}]",
                    f"[{regr_color}]{m.regression_rate:.3f}[/{regr_color}]",
                    str(m.n_season_n),
                    str(m.n_returning),
                )
            console.print(table)
        console.print()


def print_residual_persistence_report(report: ResidualPersistenceReport) -> None:
    """Print residual persistence diagnostic report."""
    console.print(
        f"[bold]Residual Persistence Diagnostic[/bold] — {report.system}/{report.version}"
        f" ({report.season_n} -> {report.season_n1}, batters)"
    )
    console.print()

    # Correlation table
    if report.stat_metrics:
        corr_table = Table(show_edge=False, pad_edge=False)
        corr_table.add_column("Stat")
        corr_table.add_column("Overall r", justify="right")
        corr_table.add_column("<200 PA", justify="right")
        corr_table.add_column("200-400", justify="right")
        corr_table.add_column("400+", justify="right")
        corr_table.add_column("N Ret", justify="right")
        for m in report.stat_metrics:
            color = "green" if m.persistence_pass else "red"
            lt200 = f"{m.residual_corr_by_bucket.get('<200', 0.0):.3f}"
            mid = f"{m.residual_corr_by_bucket.get('200-400', 0.0):.3f}"
            gt400 = f"{m.residual_corr_by_bucket.get('400+', 0.0):.3f}"
            corr_table.add_row(
                m.stat_name,
                f"[{color}]{m.residual_corr_overall:.3f}[/{color}]",
                lt200,
                mid,
                gt400,
                str(m.n_returning),
            )
        console.print("[bold]Residual Correlation (year-over-year):[/bold]")
        console.print(corr_table)
        console.print()

        # RMSE ceiling table
        rmse_table = Table(show_edge=False, pad_edge=False)
        rmse_table.add_column("Stat")
        rmse_table.add_column("RMSE Base", justify="right")
        rmse_table.add_column("RMSE Corr", justify="right")
        rmse_table.add_column("Improv %", justify="right")
        for m in report.stat_metrics:
            color = "green" if m.ceiling_pass else "red"
            rmse_table.add_row(
                m.stat_name,
                f"{m.rmse_baseline:.4f}",
                f"{m.rmse_corrected:.4f}",
                f"[{color}]{m.rmse_improvement_pct:.1f}%[/{color}]",
            )
        console.print("[bold]RMSE Improvement Ceiling:[/bold]")
        console.print(rmse_table)
        console.print()

        # Chronic performers
        for m in report.stat_metrics:
            if m.chronic_overperformers or m.chronic_underperformers:
                console.print(f"[bold]Chronic Performers — {m.stat_name}:[/bold]")
                if m.chronic_overperformers:
                    console.print("  [green]Overperformers:[/green]")
                    for p in m.chronic_overperformers:
                        console.print(
                            f"    {p.player_name}: res_N={p.residual_n:+.4f}"
                            f" res_N+1={p.residual_n1:+.4f}"
                            f" mean={p.mean_residual:+.4f}"
                            f" PA={p.pa_n:.0f}/{p.pa_n1:.0f}"
                        )
                if m.chronic_underperformers:
                    console.print("  [red]Underperformers:[/red]")
                    for p in m.chronic_underperformers:
                        console.print(
                            f"    {p.player_name}: res_N={p.residual_n:+.4f}"
                            f" res_N+1={p.residual_n1:+.4f}"
                            f" mean={p.mean_residual:+.4f}"
                            f" PA={p.pa_n:.0f}/{p.pa_n1:.0f}"
                        )
                console.print()

    # Go/No-Go summary
    s = report.summary
    console.print("[bold]Go/No-Go Summary:[/bold]")
    p_color = "green" if s.persistence_passes >= 3 else "red"
    c_color = "green" if s.ceiling_passes >= 2 else "red"
    go_color = "green" if s.go else "red"
    console.print(
        f"  Persistence passes: [{p_color}]{s.persistence_passes}/{s.persistence_total}[/{p_color}] (need 3+)"
    )
    console.print(f"  Ceiling passes:     [{c_color}]{s.ceiling_passes}/{s.ceiling_total}[/{c_color}] (need 2+)")
    verdict = "GO" if s.go else "NO-GO"
    console.print(f"  Verdict: [{go_color}][bold]{verdict}[/bold][/{go_color}]")
    console.print()


def print_residual_analysis_report(report: ResidualAnalysisReport) -> None:
    """Print residual analysis diagnostic report."""
    top_label = f", top {report.top}" if report.top else ""
    console.print(
        f"[bold]Residual Analysis[/bold] — {report.system}/{report.version}"
        f" (seasons {', '.join(str(s) for s in report.seasons)}{top_label})"
    )
    console.print()

    if report.stat_analyses:
        # Per-stat bias and heteroscedasticity table
        stat_table = Table(show_edge=False, pad_edge=False)
        stat_table.add_column("Stat")
        stat_table.add_column("Type")
        stat_table.add_column("N", justify="right")
        stat_table.add_column("Mean Resid", justify="right")
        stat_table.add_column("Std Resid", justify="right")
        stat_table.add_column("Bias Sig?", justify="center")
        stat_table.add_column("Hetero r", justify="right")
        stat_table.add_column("Hetero Sig?", justify="center")

        for a in report.stat_analyses:
            bias_color = "red" if a.bias_significant else "green"
            hetero_color = "red" if a.heteroscedasticity_significant else "green"
            stat_table.add_row(
                a.stat_name,
                a.player_type,
                str(a.n_observations),
                f"{a.mean_residual:+.4f}",
                f"{a.std_residual:.4f}",
                f"[{bias_color}]{'yes' if a.bias_significant else 'no'}[/{bias_color}]",
                f"{a.heteroscedasticity_corr:+.3f}",
                f"[{hetero_color}]{'yes' if a.heteroscedasticity_significant else 'no'}[/{hetero_color}]",
            )
        console.print(stat_table)
        console.print()

        # Calibration bins per stat
        for a in report.stat_analyses:
            if a.calibration_bins:
                console.print(f"[bold]Calibration Bins — {a.stat_name} ({a.player_type}):[/bold]")
                bin_table = Table(show_edge=False, pad_edge=False)
                bin_table.add_column("Bin Center", justify="right")
                bin_table.add_column("Mean Pred", justify="right")
                bin_table.add_column("Mean Actual", justify="right")
                bin_table.add_column("Mean Resid", justify="right")
                bin_table.add_column("Count", justify="right")
                for b in a.calibration_bins:
                    bin_table.add_row(
                        f"{b.bin_center:.4f}",
                        f"{b.mean_predicted:.4f}",
                        f"{b.mean_actual:.4f}",
                        f"{b.mean_residual:+.4f}",
                        str(b.count),
                    )
                console.print(bin_table)
                console.print()

    # Summary
    s = report.summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Bias significant: {s.n_bias_significant}/{s.n_bias_total}")
    console.print(f"  Heteroscedasticity significant: {s.n_hetero_significant}/{s.n_hetero_total}")
    if s.calibration_recommended:
        console.print("  [yellow]Calibration recommended[/yellow]")
    else:
        console.print("  [green]No calibration needed[/green]")
    console.print()


def print_value_over_adp(report: ValueOverADPReport) -> None:
    """Print a Value-Over-ADP report with buy targets, avoids, and sleepers."""
    console.print(
        f"[bold]Value-Over-ADP[/bold] — {report.system} v{report.version}"
        f" | season {report.season} | provider {report.provider}"
        f" | {report.n_matched} matched"
    )
    console.print()

    if report.buy_targets:
        console.print("[bold green]Buy Targets[/bold green] (market undervalues)")
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Delta", justify="right")
        table.add_column("Player")
        table.add_column("Type")
        table.add_column("Pos")
        table.add_column("ZAR$", justify="right")
        table.add_column("ZARRk", justify="right")
        table.add_column("ADPRk", justify="right")
        table.add_column("ADPPick", justify="right")
        for entry in report.buy_targets:
            table.add_row(
                f"[green]+{entry.rank_delta}[/green]",
                entry.player_name,
                entry.player_type,
                entry.position,
                f"${entry.zar_value:.1f}",
                str(entry.zar_rank),
                str(entry.adp_rank),
                f"{entry.adp_pick:.1f}",
            )
        console.print(table)
        console.print()

    if report.avoid_list:
        console.print("[bold red]Avoid List[/bold red] (market overvalues)")
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Delta", justify="right")
        table.add_column("Player")
        table.add_column("Type")
        table.add_column("Pos")
        table.add_column("ZAR$", justify="right")
        table.add_column("ZARRk", justify="right")
        table.add_column("ADPRk", justify="right")
        table.add_column("ADPPick", justify="right")
        for entry in report.avoid_list:
            table.add_row(
                f"[red]{entry.rank_delta}[/red]",
                entry.player_name,
                entry.player_type,
                entry.position,
                f"${entry.zar_value:.1f}",
                str(entry.zar_rank),
                str(entry.adp_rank),
                f"{entry.adp_pick:.1f}",
            )
        console.print(table)
        console.print()

    if report.unranked_valuable:
        console.print("[bold yellow]Unranked Sleepers[/bold yellow] (ZAR value but no ADP)")
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Player")
        table.add_column("Type")
        table.add_column("Pos")
        table.add_column("ZAR$", justify="right")
        table.add_column("ZARRk", justify="right")
        for entry in report.unranked_valuable:
            table.add_row(
                entry.player_name,
                entry.player_type,
                entry.position,
                f"${entry.zar_value:.1f}",
                str(entry.zar_rank),
            )
        console.print(table)
        console.print()

    if not report.buy_targets and not report.avoid_list and not report.unranked_valuable:
        console.print("No discrepancies found.")


def print_adp_accuracy_report(report: ADPAccuracyReport) -> None:
    """Print ADP accuracy evaluation report."""
    n_seasons = len(report.seasons)
    has_comparison = report.comparison is not None and len(report.comparison) > 0

    if n_seasons == 1:
        result = report.adp_results[0]

        if has_comparison:
            assert report.comparison is not None  # noqa: S101 - type narrowing
            sys_result = report.comparison[0]
            console.print(f"[bold]ADP Accuracy[/bold] — season {result.season} | provider {report.provider}")
            console.print()

            table = Table(show_edge=False, pad_edge=False)
            table.add_column("Metric")
            table.add_column("ADP", justify="right")
            table.add_column(f"{sys_result.system}/{sys_result.version}", justify="right")

            table.add_row("Matched", str(result.n_matched), str(sys_result.n_matched))
            table.add_row(
                "Spearman rho",
                f"{result.rank_correlation:.4f}",
                f"{sys_result.rank_correlation:.4f}",
            )
            table.add_row("Value RMSE", f"${result.value_rmse:.2f}", f"${sys_result.value_rmse:.2f}")
            table.add_row("Value MAE", f"${result.value_mae:.2f}", f"${sys_result.value_mae:.2f}")
            for n in sorted(result.top_n_precision):
                adp_pct = result.top_n_precision[n] * 100
                sys_pct = sys_result.top_n_precision.get(n, 0.0) * 100
                table.add_row(f"Top-{n} precision", f"{adp_pct:.1f}%", f"{sys_pct:.1f}%")
            console.print(table)
        else:
            console.print(
                f"[bold]ADP Accuracy[/bold] — season {result.season}"
                f" | provider {report.provider}"
                f" | {result.n_matched} matched"
            )
            console.print()
            console.print(f"  Spearman rank correlation: {result.rank_correlation:.4f}")
            console.print(f"  Value RMSE: ${result.value_rmse:.2f}")
            console.print(f"  Value MAE: ${result.value_mae:.2f}")
            for n in sorted(result.top_n_precision):
                pct = result.top_n_precision[n] * 100
                console.print(f"  Top-{n} precision: {pct:.1f}%")
    else:
        console.print(f"[bold]ADP Accuracy[/bold] — {n_seasons} seasons | provider {report.provider}")
        console.print()

        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Season")
        table.add_column("Matched", justify="right")
        table.add_column("Spearman", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("MAE", justify="right")
        for n in sorted(report.mean_top_n_precision):
            table.add_column(f"Top{n}", justify="right")

        for result in report.adp_results:
            row: list[str] = [
                str(result.season),
                str(result.n_matched),
                f"{result.rank_correlation:.4f}" if result.n_matched >= 3 else "—",
                f"${result.value_rmse:.2f}",
                f"${result.value_mae:.2f}",
            ]
            for n in sorted(result.top_n_precision):
                pct = result.top_n_precision[n] * 100
                row.append(f"{pct:.1f}%")
            table.add_row(*row)

        # Aggregate row
        agg_row: list[str] = [
            "[bold]Mean[/bold]",
            "",
            f"[bold]{report.mean_rank_correlation:.4f}[/bold]",
            f"[bold]${report.mean_value_rmse:.2f}[/bold]",
            "",
        ]
        for n in sorted(report.mean_top_n_precision):
            pct = report.mean_top_n_precision[n] * 100
            agg_row.append(f"[bold]{pct:.1f}%[/bold]")
        table.add_row(*agg_row)

        console.print(table)


def print_adp_movers_report(report: ADPMoversReport) -> None:
    """Print an ADP movers report with risers, fallers, new entries, and dropped."""
    console.print(
        f"[bold]ADP Movers[/bold] — season {report.season}"
        f" | provider {report.provider}"
        f" | {report.previous_as_of} -> {report.current_as_of}"
    )
    console.print()

    if report.risers:
        console.print("[bold green]Risers[/bold green]")
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Delta", justify="right")
        table.add_column("Player")
        table.add_column("Pos")
        table.add_column("Current", justify="right")
        table.add_column("Previous", justify="right")
        for m in report.risers:
            table.add_row(
                f"[green]+{m.rank_delta}[/green]",
                m.player_name,
                m.position,
                str(m.current_rank),
                str(m.previous_rank),
            )
        console.print(table)
        console.print()

    if report.fallers:
        console.print("[bold red]Fallers[/bold red]")
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Delta", justify="right")
        table.add_column("Player")
        table.add_column("Pos")
        table.add_column("Current", justify="right")
        table.add_column("Previous", justify="right")
        for m in report.fallers:
            table.add_row(
                f"[red]{m.rank_delta}[/red]",
                m.player_name,
                m.position,
                str(m.current_rank),
                str(m.previous_rank),
            )
        console.print(table)
        console.print()

    if report.new_entries:
        console.print("[bold yellow]New Entries[/bold yellow]")
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Player")
        table.add_column("Pos")
        table.add_column("Rank", justify="right")
        for m in report.new_entries:
            table.add_row(m.player_name, m.position, str(m.current_rank))
        console.print(table)
        console.print()

    if report.dropped_entries:
        console.print("[bold dim]Dropped[/bold dim]")
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Player")
        table.add_column("Pos")
        table.add_column("Last Rank", justify="right")
        for m in report.dropped_entries:
            table.add_row(m.player_name, m.position, str(m.previous_rank))
        console.print(table)
        console.print()

    if not report.risers and not report.fallers and not report.new_entries and not report.dropped_entries:
        console.print("No movers found.")


def print_projection_confidence(report: ConfidenceReport) -> None:
    """Print a projection confidence report showing cross-system agreement."""
    if not report.players:
        console.print("No players with sufficient projection systems.")
        return

    console.print(
        f"[bold]Projection Confidence[/bold] — season {report.season}"
        f" | {len(report.systems)} systems: {', '.join(report.systems)}"
        f" | {len(report.players)} players"
    )
    console.print()

    # Collect all stat keys across players for column headers
    stat_keys: list[str] = []
    seen: set[str] = set()
    for player in report.players:
        for spread in player.spreads:
            if spread.stat not in seen:
                stat_keys.append(spread.stat)
                seen.add(spread.stat)

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Player")
    table.add_column("Type")
    table.add_column("Pos")
    table.add_column("CV", justify="right")
    table.add_column("Agreement")
    for stat in stat_keys:
        table.add_column(stat, justify="right")

    for player in report.players:
        agreement_color = {"high": "green", "medium": "yellow", "low": "red"}.get(player.agreement_level, "")
        agreement_str = f"[{agreement_color}]{player.agreement_level}[/{agreement_color}]"

        spread_by_stat = {s.stat: s for s in player.spreads}
        cells: list[str] = [
            player.player_name,
            player.player_type,
            player.position,
            f"{player.overall_cv:.3f}",
            agreement_str,
        ]
        for stat in stat_keys:
            spread = spread_by_stat.get(stat)
            if spread is not None:
                cells.append(f"{spread.min_value:.0f}-{spread.max_value:.0f}")
            else:
                cells.append("")
        table.add_row(*cells)

    console.print(table)


_CLASSIFICATION_ORDER: list[VarianceClassification] = [
    VarianceClassification.UPSIDE_GAMBLE,
    VarianceClassification.HIDDEN_UPSIDE,
    VarianceClassification.SAFE_CONSENSUS,
    VarianceClassification.KNOWN_QUANTITY,
    VarianceClassification.RISKY_AVOID,
]

_CLASSIFICATION_STYLES: dict[VarianceClassification, str] = {
    VarianceClassification.UPSIDE_GAMBLE: "bold green",
    VarianceClassification.HIDDEN_UPSIDE: "green",
    VarianceClassification.SAFE_CONSENSUS: "dim",
    VarianceClassification.KNOWN_QUANTITY: "dim",
    VarianceClassification.RISKY_AVOID: "bold red",
}

_CLASSIFICATION_LABELS: dict[VarianceClassification, str] = {
    VarianceClassification.UPSIDE_GAMBLE: "Upside Gamble — Draft Targets",
    VarianceClassification.HIDDEN_UPSIDE: "Hidden Upside — Draft Targets",
    VarianceClassification.SAFE_CONSENSUS: "Safe Consensus",
    VarianceClassification.KNOWN_QUANTITY: "Known Quantity",
    VarianceClassification.RISKY_AVOID: "Risky Avoid — Fade",
}


def print_variance_targets(classified: list[ClassifiedPlayer]) -> None:
    """Print variance-classified players grouped by classification bucket."""
    if not classified:
        console.print("No classified players.")
        return

    console.print(f"[bold]Variance Targets[/bold] — {len(classified)} players classified")
    console.print()

    # Group by classification
    by_class: dict[VarianceClassification, list[ClassifiedPlayer]] = {}
    for cp in classified:
        by_class.setdefault(cp.classification, []).append(cp)

    for cls in _CLASSIFICATION_ORDER:
        group = by_class.get(cls)
        if not group:
            continue

        style = _CLASSIFICATION_STYLES[cls]
        label = _CLASSIFICATION_LABELS[cls]
        console.print(f"[{style}]{label}[/{style}] ({len(group)})")

        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Player")
        table.add_column("Type")
        table.add_column("Pos")
        table.add_column("Agreement")
        table.add_column("ValRk", justify="right")
        table.add_column("ADPRk", justify="right")
        table.add_column("RR Score", justify="right")

        for cp in group:
            rr = cp.risk_reward_score
            rr_color = "green" if rr > 0 else "red" if rr < 0 else ""
            rr_str = f"[{rr_color}]{rr:+.1f}[/{rr_color}]" if rr_color else f"{rr:+.1f}"
            table.add_row(
                cp.player.player_name,
                cp.player.player_type,
                cp.player.position,
                cp.player.agreement_level,
                str(cp.value_rank),
                str(cp.adp_rank) if cp.adp_rank is not None else "—",
                rr_str,
            )

        console.print(table)
        console.print()


def print_system_disagreements(player: PlayerConfidence, projections: list[Projection]) -> None:
    """Print per-system stat comparison for a single player."""
    console.print(f"[bold]System Disagreements[/bold] — {player.player_name} ({player.position})")
    console.print()

    if not player.spreads:
        console.print("No stat spreads available.")
        return

    # Sort spreads by CV descending (widest disagreement first)
    sorted_spreads = sorted(player.spreads, key=lambda s: s.cv, reverse=True)

    # Collect all systems from projections
    systems = sorted({p.system for p in projections})

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Stat")
    table.add_column("CV", justify="right")
    for sys in systems:
        table.add_column(sys, justify="right")

    # Build a lookup: system → stat_json
    system_stats: dict[str, dict[str, float]] = {}
    for proj in projections:
        if proj.system not in system_stats:
            system_stats[proj.system] = {}
        for k, v in proj.stat_json.items():
            if isinstance(v, int | float):
                system_stats[proj.system][k] = float(v)

    for spread in sorted_spreads:
        # Get per-system values for this stat
        values: dict[str, float] = {}
        for sys in systems:
            val = system_stats.get(sys, {}).get(spread.stat)
            if val is not None:
                values[sys] = val

        if not values:
            continue

        min_val = min(values.values())
        max_val = max(values.values())

        cells: list[str] = [spread.stat, f"{spread.cv:.3f}"]
        for sys in systems:
            val = values.get(sys)
            if val is None:
                cells.append("—")
            elif val == min_val and min_val != max_val:
                cells.append(f"[dim]{val:.1f}[/dim]")
            elif val == max_val and min_val != max_val:
                cells.append(f"[bold]{val:.1f}[/bold]")
            else:
                cells.append(f"{val:.1f}")
        table.add_row(*cells)

    console.print(table)
