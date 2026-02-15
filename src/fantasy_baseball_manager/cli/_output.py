import json

from rich.console import Console
from rich.table import Table

from fantasy_baseball_manager.domain.evaluation import ComparisonResult, SystemMetrics
from fantasy_baseball_manager.domain.load_log import LoadLog
from fantasy_baseball_manager.domain.model_run import ModelRunRecord
from fantasy_baseball_manager.domain.projection import PlayerProjection, SystemSummary
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
    table.add_column("N", justify="right")
    for stat_name in sorted(metrics.metrics):
        m = metrics.metrics[stat_name]
        table.add_row(stat_name, f"{m.rmse:.4f}", f"{m.mae:.4f}", f"{m.correlation:.4f}", str(m.n))
    console.print(table)


def print_comparison_result(result: ComparisonResult) -> None:
    """Print comparison table across systems."""
    console.print(f"Comparison — season [bold]{result.season}[/bold]")
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Stat")
    for sys_metrics in result.systems:
        table.add_column(f"{sys_metrics.system}/{sys_metrics.version}", justify="right")
    for stat_name in result.stats:
        values: list[str] = []
        for sys_metrics in result.systems:
            m = sys_metrics.metrics.get(stat_name)
            values.append(f"{m.rmse:.4f}" if m else "—")
        table.add_row(stat_name, *values)
    console.print(table)


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
