import json

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from fantasy_baseball_manager.domain.evaluation import ComparisonResult, StratifiedComparisonResult, SystemMetrics
from fantasy_baseball_manager.domain.load_log import LoadLog
from fantasy_baseball_manager.domain.performance_delta import PlayerStatDelta
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
