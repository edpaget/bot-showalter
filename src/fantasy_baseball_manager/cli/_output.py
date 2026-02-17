import json

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from fantasy_baseball_manager.domain.evaluation import ComparisonResult, StratifiedComparisonResult, SystemMetrics
from fantasy_baseball_manager.domain.load_log import LoadLog
from fantasy_baseball_manager.domain.performance_delta import PlayerStatDelta
from fantasy_baseball_manager.domain.residual_persistence import ResidualPersistenceReport
from fantasy_baseball_manager.domain.talent_quality import TrueTalentQualityReport
from fantasy_baseball_manager.domain.model_run import ModelRunRecord
from fantasy_baseball_manager.domain.projection import PlayerProjection, SystemSummary
from fantasy_baseball_manager.domain.valuation import PlayerValuation, ValuationEvalResult
from fantasy_baseball_manager.services.dataset_catalog import DatasetInfo
from fantasy_baseball_manager.features.types import AnyFeature, DeltaFeature, DerivedTransformFeature, TransformFeature
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    PredictResult,
    PrepareResult,
    TrainResult,
    TuneResult,
)

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)


def print_error(message: str) -> None:
    err_console.print(f"[red bold]Error:[/red bold] {message}")


def print_prepare_result(result: PrepareResult) -> None:
    console.print(f"[bold green]Prepared[/bold green] model [bold]'{result.model_name}'[/bold]")
    console.print(f"  Rows processed: {result.rows_processed}")
    console.print(f"  Artifacts: {result.artifacts_path}")


def print_train_result(result: TrainResult) -> None:
    console.print(f"[bold green]Trained[/bold green] model [bold]'{result.model_name}'[/bold]")
    if result.metrics:
        for name, value in result.metrics.items():
            console.print(f"  {name}: {value}")
    console.print(f"  Artifacts: {result.artifacts_path}")


def print_predict_result(result: PredictResult) -> None:
    console.print(f"[bold green]Predictions[/bold green] from model [bold]'{result.model_name}'[/bold]")
    console.print(f"  {len(result.predictions)} predictions saved to database")


def print_ablation_result(result: AblationResult) -> None:
    console.print(f"Ablation results for model [bold]'{result.model_name}'[/bold]")
    if result.feature_impacts:
        for feature, impact in sorted(result.feature_impacts.items(), key=lambda x: -abs(x[1])):
            color = "green" if impact > 0 else "red"
            console.print(f"  {feature}: [{color}]{impact:+.4f}[/{color}]")


def print_tune_result(result: TuneResult) -> None:
    """Print tuning results with best params in TOML-ready format."""
    console.print(f"[bold green]Tuning complete[/bold green] for model [bold]'{result.model_name}'[/bold]")
    console.print()

    # Batter results
    console.print("[bold]Batter best params:[/bold]")
    for key, value in sorted(result.batter_params.items()):
        console.print(f"  {key} = {value!r}")
    console.print("[bold]Batter CV RMSE:[/bold]")
    for target, rmse in sorted(result.batter_cv_rmse.items()):
        console.print(f"  {target}: {rmse:.4f}")
    console.print()

    # Pitcher results
    console.print("[bold]Pitcher best params:[/bold]")
    for key, value in sorted(result.pitcher_params.items()):
        console.print(f"  {key} = {value!r}")
    console.print("[bold]Pitcher CV RMSE:[/bold]")
    for target, rmse in sorted(result.pitcher_cv_rmse.items()):
        console.print(f"  {target}: {rmse:.4f}")
    console.print()

    # TOML snippet
    console.print("[bold]TOML snippet (copy into fbm.toml):[/bold]")
    console.print("[dim]# Batter params[/dim]")
    for key, value in sorted(result.batter_params.items()):
        if value is None:
            console.print(f'# {key} = "None"  # unlimited')
        elif isinstance(value, float):
            console.print(f"{key} = {value}")
        else:
            console.print(f"{key} = {value}")
    console.print()
    console.print("[dim]# Pitcher params[/dim]")
    for key, value in sorted(result.pitcher_params.items()):
        if value is None:
            console.print(f'# {key} = "None"  # unlimited')
        elif isinstance(value, float):
            console.print(f"{key} = {value}")
        else:
            console.print(f"{key} = {value}")


def print_import_result(log: LoadLog) -> None:
    console.print(f"[bold green]Import complete:[/bold green] {log.rows_loaded} projections loaded")
    console.print(f"  Source: {log.source_detail}")
    console.print(f"  Status: {log.status}")


def print_ingest_result(log: LoadLog) -> None:
    console.print(f"[bold green]Ingest complete:[/bold green] {log.rows_loaded} rows loaded into {log.target_table}")
    console.print(f"  Source: {log.source_detail}")
    console.print(f"  Status: {log.status}")
    if log.error_message:
        console.print(f"  [red]Error: {log.error_message}[/red]")


def print_system_metrics(metrics: SystemMetrics) -> None:
    """Print evaluation results in tabular format."""
    console.print(f"Evaluation: [bold]{metrics.system}[/bold] v{metrics.version} [dim]({metrics.source_type})[/dim]")
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Stat")
    table.add_column("RMSE", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("r", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("N", justify="right")
    for stat_name in sorted(metrics.metrics):
        m = metrics.metrics[stat_name]
        table.add_row(
            stat_name, f"{m.rmse:.4f}", f"{m.mae:.4f}", f"{m.correlation:.4f}", f"{m.r_squared:.4f}", str(m.n)
        )
    console.print(table)


def print_comparison_result(result: ComparisonResult) -> None:
    """Print comparison table across systems."""
    console.print(f"Comparison — season [bold]{result.season}[/bold]")
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Stat")
    for sys_metrics in result.systems:
        label = f"{sys_metrics.system}/{sys_metrics.version}"
        table.add_column(f"{label} RMSE", justify="right")
        table.add_column(f"{label} R²", justify="right")
    for stat_name in result.stats:
        values: list[str] = []
        for sys_metrics in result.systems:
            m = sys_metrics.metrics.get(stat_name)
            values.append(f"{m.rmse:.4f}" if m else "—")
            values.append(f"{m.r_squared:.4f}" if m else "—")
        table.add_row(stat_name, *values)
    console.print(table)


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


def print_run_list(records: list[ModelRunRecord]) -> None:
    """Print a table of model runs."""
    if not records:
        console.print("No runs found.")
        return
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("System")
    table.add_column("Version")
    table.add_column("Operation")
    table.add_column("Created")
    table.add_column("Tags")
    for r in records:
        tags_str = ", ".join(f"{k}={v}" for k, v in r.tags_json.items()) if r.tags_json else ""
        table.add_row(r.system, r.version, r.operation, r.created_at, tags_str)
    console.print(table)


def print_run_detail(record: ModelRunRecord) -> None:
    """Print full details of a model run."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("System", record.system)
    table.add_row("Version", record.version)
    table.add_row("Operation", record.operation)
    table.add_row("Created", record.created_at)
    table.add_row("Git Commit", record.git_commit or "N/A")
    table.add_row("Artifact Type", record.artifact_type)
    table.add_row("Artifact Path", record.artifact_path or "N/A")
    if record.config_json:
        table.add_row("Config", json.dumps(record.config_json, indent=2))
    if record.metrics_json:
        table.add_row("Metrics", json.dumps(record.metrics_json, indent=2))
    if record.tags_json:
        tags_str = ", ".join(f"{k}={v}" for k, v in record.tags_json.items())
        table.add_row("Tags", tags_str)
    console.print(table)


def print_features(model_name: str, features: tuple[AnyFeature, ...]) -> None:
    console.print(f"Features for model [bold]'{model_name}'[/bold] ({len(features)} features):")
    table = Table(show_header=True, show_edge=False, pad_edge=False)
    table.add_column("Name")
    table.add_column("Details")
    for f in features:
        if isinstance(f, DeltaFeature):
            table.add_row(f.name, f"delta({f.left.name} - {f.right.name})")
        elif isinstance(f, TransformFeature):
            outputs = ", ".join(f.outputs)
            table.add_row(f.name, f"{f.source.value} transform → {outputs}")
        elif isinstance(f, DerivedTransformFeature):
            outputs = ", ".join(f.outputs)
            table.add_row(f.name, f"derived transform → {outputs}")
        elif f.computed:
            table.add_row(f.name, f"{f.source.value} computed={f.computed}")
        else:
            detail = f"{f.source.value}.{f.column}"
            if f.lag:
                detail += f" lag={f.lag}"
            if f.system:
                detail += f" system={f.system}"
            table.add_row(f.name, detail)
    console.print(table)


_METADATA_KEYS = {"_components", "_mode", "_pt_system", "rates"}


def print_player_projections(projections: list[PlayerProjection]) -> None:
    """Print player projection results."""
    if not projections:
        console.print("No projections found.")
        return
    for proj in projections:
        console.print(
            f"[bold]{proj.player_name}[/bold] — {proj.system} v{proj.version}"
            f" [dim]({proj.source_type}, {proj.player_type})[/dim]"
        )
        # Lineage: ensemble sources
        components = proj.stats.get("_components")
        if isinstance(components, dict):
            mode = proj.stats.get("_mode", "")
            parts = [f"{sys} {int(w * 100)}%" for sys, w in components.items()]
            console.print(f"  Sources: {', '.join(parts)} ({mode})")
        # Lineage: composite PT source
        pt_system = proj.stats.get("_pt_system")
        if isinstance(pt_system, str):
            console.print(f"  PT source: {pt_system}")
        # Stats table, filtering out metadata keys
        table = Table(show_header=False, show_edge=False, pad_edge=False, box=None)
        table.add_column("Stat")
        table.add_column("Value", justify="right")
        for stat_name in sorted(proj.stats):
            if stat_name in _METADATA_KEYS or stat_name.startswith("_"):
                continue
            value = proj.stats[stat_name]
            if isinstance(value, float):
                table.add_row(stat_name, f"{value:.3f}")
            else:
                table.add_row(stat_name, str(value))
        console.print(table)


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


def print_system_summaries(summaries: list[SystemSummary]) -> None:
    """Print a table of available projection systems."""
    if not summaries:
        console.print("No projection systems found for this season.")
        return
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("System")
    table.add_column("Version")
    table.add_column("Source")
    table.add_column("Batters", justify="right")
    table.add_column("Pitchers", justify="right")
    table.add_column("Total", justify="right")
    for s in summaries:
        total = s.batter_count + s.pitcher_count
        table.add_row(s.system, s.version, s.source_type, str(s.batter_count), str(s.pitcher_count), str(total))
    console.print(table)


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


def print_player_valuations(valuations: list[PlayerValuation]) -> None:
    """Print player valuation results with category z-score breakdown."""
    if not valuations:
        console.print("No valuations found.")
        return
    for val in valuations:
        console.print(
            f"[bold]{val.player_name}[/bold] — {val.system} v{val.version}"
            f" [dim]({val.player_type}, {val.position})[/dim]"
        )
        console.print(f"  Projection: {val.projection_system} v{val.projection_version}")
        console.print(f"  Value: [bold]${val.value:.1f}[/bold]  Rank: {val.rank}")
        if val.category_scores:
            table = Table(show_header=False, show_edge=False, pad_edge=False, box=None)
            table.add_column("Category")
            table.add_column("Z-Score", justify="right")
            for cat in sorted(val.category_scores):
                z = val.category_scores[cat]
                table.add_row(cat, f"{z:.2f}")
            console.print(table)


def print_valuation_rankings(rankings: list[PlayerValuation]) -> None:
    """Print a valuation rankings leaderboard table."""
    if not rankings:
        console.print("No valuations found.")
        return
    category_names = sorted(rankings[0].category_scores) if rankings[0].category_scores else []
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Player")
    table.add_column("Type")
    table.add_column("Pos")
    table.add_column("Value", justify="right")
    table.add_column("System")
    for cat in category_names:
        table.add_column(cat, justify="right")
    for val in rankings:
        row: list[str] = [
            str(val.rank),
            val.player_name,
            val.player_type,
            val.position,
            f"${val.value:.1f}",
            val.system,
        ]
        for cat in category_names:
            z = val.category_scores.get(cat, 0.0)
            row.append(f"{z:.2f}")
        table.add_row(*row)
    console.print(table)


def print_valuation_eval_result(result: ValuationEvalResult, top: int | None = None) -> None:
    """Print valuation evaluation results with metrics and per-player breakdown."""
    if result.n == 0:
        console.print("No matched players found.")
        return

    console.print(
        f"Valuation evaluation: [bold]{result.system}[/bold] v{result.version}"
        f" — season {result.season} ({result.n} matched players)"
    )
    console.print(f"  Value MAE: [bold]{result.value_mae:.2f}[/bold]")
    console.print(f"  Spearman rank correlation: [bold]{result.rank_correlation:.4f}[/bold]")
    console.print()

    players = result.players
    if top is not None:
        players = players[:top]

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Player")
    table.add_column("Type")
    table.add_column("Predicted$", justify="right")
    table.add_column("Actual$", justify="right")
    table.add_column("Surplus$", justify="right")
    table.add_column("PredRank", justify="right")
    table.add_column("ActRank", justify="right")
    for p in players:
        surplus_str = f"{p.surplus:+.1f}"
        table.add_row(
            p.player_name,
            p.player_type,
            f"${p.predicted_value:.1f}",
            f"${p.actual_value:.1f}",
            surplus_str,
            str(p.predicted_rank),
            str(p.actual_rank),
        )
    console.print(table)


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
