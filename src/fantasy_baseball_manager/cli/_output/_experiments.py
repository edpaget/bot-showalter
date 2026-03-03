import json
from statistics import mean
from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        Experiment,
        ExplorationSummary,
        FeatureCheckpoint,
    )
    from fantasy_baseball_manager.services import FeatureSetComparisonResult


def print_experiment_search_results(experiments: list[Experiment], target: str | None = None) -> None:
    if not experiments:
        console.print("[dim]No experiments found.[/dim]")
        return

    table = Table(title="Experiment Search Results")
    table.add_column("ID", style="bold")
    table.add_column("Timestamp")
    table.add_column("Model")
    table.add_column("Hypothesis")
    table.add_column("Delta %", justify="right")
    table.add_column("Tags")

    for exp in experiments:
        hypothesis = exp.hypothesis[:40] + "..." if len(exp.hypothesis) > 40 else exp.hypothesis
        if target and target in exp.target_results:
            delta = exp.target_results[target].delta_pct
        elif exp.target_results:
            delta = mean(tr.delta_pct for tr in exp.target_results.values())
        else:
            delta = 0.0
        style = "green" if delta < 0 else "red"
        table.add_row(
            str(exp.id),
            exp.timestamp,
            exp.model,
            hypothesis,
            f"[{style}]{delta:+.2f}%[/{style}]",
            ", ".join(exp.tags),
        )
    console.print(table)


def print_experiment_detail(experiment: Experiment) -> None:
    console.print(f"[bold]Experiment #{experiment.id}[/bold]")
    console.print()

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column("Field", style="bold")
    info.add_column("Value")
    info.add_row("Timestamp", experiment.timestamp)
    info.add_row("Model", experiment.model)
    info.add_row("Player type", experiment.player_type)
    info.add_row("Hypothesis", experiment.hypothesis)
    info.add_row("Conclusion", experiment.conclusion)
    info.add_row("Features added", ", ".join(experiment.feature_diff.get("added", [])) or "—")
    info.add_row("Features removed", ", ".join(experiment.feature_diff.get("removed", [])) or "—")
    info.add_row("Train seasons", ", ".join(str(s) for s in experiment.seasons.get("train", [])))
    info.add_row("Holdout seasons", ", ".join(str(s) for s in experiment.seasons.get("holdout", [])))
    info.add_row("Params", json.dumps(experiment.params, indent=2))
    info.add_row("Tags", ", ".join(experiment.tags) or "—")
    if experiment.parent_id is not None:
        info.add_row("Parent", str(experiment.parent_id))
    console.print(info)

    if experiment.target_results:
        console.print()
        tr_table = Table(title="Target Results")
        tr_table.add_column("Target")
        tr_table.add_column("RMSE", justify="right")
        tr_table.add_column("Baseline", justify="right")
        tr_table.add_column("Delta", justify="right")
        tr_table.add_column("Delta %", justify="right")
        for target, tr in sorted(experiment.target_results.items()):
            style = "green" if tr.delta_pct < 0 else "red"
            tr_table.add_row(
                target,
                f"{tr.rmse:.4f}",
                f"{tr.baseline_rmse:.4f}",
                f"[{style}]{tr.delta:+.4f}[/{style}]",
                f"[{style}]{tr.delta_pct:+.2f}%[/{style}]",
            )
        console.print(tr_table)


def print_experiment_summary(summary: ExplorationSummary) -> None:
    console.print(f"[bold]Exploration Summary: {summary.model} ({summary.player_type})[/bold]")
    console.print(f"Total experiments: {summary.total_experiments}")
    console.print()

    if summary.features_tested:
        feat_table = Table(title="Features Tested")
        feat_table.add_column("Feature")
        feat_table.add_column("Times Tested", justify="right")
        feat_table.add_column("Best Delta %", justify="right")
        feat_table.add_column("Best Experiment", justify="right")
        for f in sorted(summary.features_tested, key=lambda x: x.best_delta_pct):
            style = "green" if f.best_delta_pct < 0 else "red"
            feat_table.add_row(
                f.feature,
                str(f.times_tested),
                f"[{style}]{f.best_delta_pct:+.2f}%[/{style}]",
                f"#{f.best_experiment_id}",
            )
        console.print(feat_table)
        console.print()

    if summary.targets_explored:
        tgt_table = Table(title="Targets Explored")
        tgt_table.add_column("Target")
        tgt_table.add_column("Experiments", justify="right")
        tgt_table.add_column("Best RMSE", justify="right")
        tgt_table.add_column("Best Delta %", justify="right")
        tgt_table.add_column("Best Experiment", justify="right")
        for t in sorted(summary.targets_explored, key=lambda x: x.best_delta_pct):
            style = "green" if t.best_delta_pct < 0 else "red"
            tgt_table.add_row(
                t.target,
                str(t.experiments_count),
                f"{t.best_rmse:.4f}",
                f"[{style}]{t.best_delta_pct:+.2f}%[/{style}]",
                f"#{t.best_experiment_id}",
            )
        console.print(tgt_table)
        console.print()

    if summary.best_experiment_id is not None:
        style = "green" if (summary.best_experiment_delta_pct or 0) < 0 else "red"
        console.print(
            f"Overall best: [bold]#{summary.best_experiment_id}[/bold] "
            f"([{style}]{summary.best_experiment_delta_pct:+.2f}%[/{style}] avg delta)"
        )


def print_checkpoint_list(checkpoints: list[FeatureCheckpoint]) -> None:
    """Print a table of checkpoints."""
    if not checkpoints:
        console.print("[dim]No checkpoints found.[/dim]")
        return

    table = Table(title="Feature Checkpoints")
    table.add_column("Name", style="bold")
    table.add_column("Model")
    table.add_column("Player Type")
    table.add_column("Features", justify="right")
    table.add_column("Experiment", justify="right")
    table.add_column("Created")

    for cp in checkpoints:
        table.add_row(
            cp.name,
            cp.model,
            cp.player_type,
            str(len(cp.feature_columns)),
            f"#{cp.experiment_id}",
            cp.created_at,
        )

    console.print(table)


def print_checkpoint_detail(checkpoint: FeatureCheckpoint) -> None:
    """Print full details of a single checkpoint."""
    console.print(f"[bold]Checkpoint: {checkpoint.name}[/bold]")
    console.print()

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column("Field", style="bold")
    info.add_column("Value")
    info.add_row("Model", checkpoint.model)
    info.add_row("Player type", checkpoint.player_type)
    info.add_row("Experiment", f"#{checkpoint.experiment_id}")
    info.add_row("Created", checkpoint.created_at)
    info.add_row("Features", ", ".join(checkpoint.feature_columns) or "—")
    info.add_row("Params", json.dumps(checkpoint.params, indent=2))
    if checkpoint.notes:
        info.add_row("Notes", checkpoint.notes)
    console.print(info)

    if checkpoint.target_results:
        console.print()
        tr_table = Table(title="Target Results at Checkpoint")
        tr_table.add_column("Target")
        tr_table.add_column("RMSE", justify="right")
        tr_table.add_column("Baseline", justify="right")
        tr_table.add_column("Delta", justify="right")
        tr_table.add_column("Delta %", justify="right")
        for target, tr in sorted(checkpoint.target_results.items()):
            style = "green" if tr.delta_pct < 0 else "red"
            tr_table.add_row(
                target,
                f"{tr.rmse:.4f}",
                f"{tr.baseline_rmse:.4f}",
                f"[{style}]{tr.delta:+.4f}[/{style}]",
                f"[{style}]{tr.delta_pct:+.2f}%[/{style}]",
            )
        console.print(tr_table)


def print_compare_features_result(result: FeatureSetComparisonResult) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Target")
    table.add_column("Set A RMSE", justify="right")
    table.add_column("Set B RMSE", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Delta %", justify="right")

    for d in result.deltas:
        color = "green" if d.delta < 0 else "red" if d.delta > 0 else ""
        delta_str = f"[{color}]{d.delta:+.4f}[/{color}]" if color else f"{d.delta:+.4f}"
        pct_str = f"[{color}]{d.delta_pct:+.1f}%[/{color}]" if color else f"{d.delta_pct:+.1f}%"
        table.add_row(
            d.target,
            f"{d.baseline_rmse:.4f}",
            f"{d.candidate_rmse:.4f}",
            delta_str,
            pct_str,
        )

    console.print(table)
    verdict_color = "green" if result.n_improved > 0 else "yellow"
    console.print(
        f"  [{verdict_color}]Set B wins {result.n_improved}/{result.n_total} targets[/{verdict_color}]"
        f"  (avg delta: {result.avg_delta_pct:+.1f}%)"
    )
    if result.n_folds > 1:
        console.print(f"  Averaged over {result.n_folds} folds")
