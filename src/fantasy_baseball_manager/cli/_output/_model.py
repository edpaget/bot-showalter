from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.models import (
        AblationResult,
        PredictResult,
        PrepareResult,
        TrainResult,
        TuneResult,
        ValidationResult,
    )


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
    if result.group_impacts:
        console.print("  [bold]Feature Groups:[/bold]")
        for group_name, impact in sorted(result.group_impacts.items(), key=lambda x: -abs(x[1])):
            se = result.group_standard_errors.get(group_name, 0.0)
            ci_lo = impact - 2 * se
            ci_hi = impact + 2 * se
            color = "green" if impact > 0 else "red"
            prune = " [yellow]\\[GROUP PRUNE][/yellow]" if impact + 2 * se <= 0 else ""
            members = result.group_members.get(group_name, [])
            n_features = len(members)
            console.print(
                f"    {group_name} ({n_features} features): [{color}]{impact:+.4f}[/{color}]"
                f" (SE: {se:.4f}, 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]){prune}"
            )
            if members:
                console.print(f"      {', '.join(members)}")
    if result.feature_impacts:
        has_se = bool(result.feature_standard_errors)
        for feature, impact in sorted(result.feature_impacts.items(), key=lambda x: -abs(x[1])):
            color = "green" if impact > 0 else "red"
            if has_se:
                se = result.feature_standard_errors.get(feature, 0.0)
                ci_lo = impact - 2 * se
                ci_hi = impact + 2 * se
                prune = " [yellow]\\[PRUNE][/yellow]" if impact + 2 * se <= 0 else ""
                console.print(
                    f"  {feature}: [{color}]{impact:+.4f}[/{color}]"
                    f" (SE: {se:.4f}, 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]){prune}"
                )
            else:
                console.print(f"  {feature}: [{color}]{impact:+.4f}[/{color}]")
    if result.validation_results:
        console.print()
        console.print("[bold]Pruning Validation:[/bold]")
        for _, vr in sorted(result.validation_results.items()):
            _print_validation_result(vr)


def _print_validation_result(vr: ValidationResult) -> None:
    verdict = "GO" if vr.go else "NO-GO"
    verdict_color = "green" if vr.go else "red"
    console.print(
        f"  {vr.player_type}: [{verdict_color}][bold]{verdict}[/bold][/{verdict_color}]"
        f" ({vr.n_improved} improved, {vr.n_degraded} degraded,"
        f" max degradation: {vr.max_degradation_pct:.1f}%)"
    )
    if vr.pruned_features:
        console.print(f"    Pruned: {', '.join(vr.pruned_features)}")
    if vr.comparisons:
        table = Table(show_edge=False, pad_edge=False)
        table.add_column("Target")
        table.add_column("Full RMSE", justify="right")
        table.add_column("Pruned RMSE", justify="right")
        table.add_column("Delta %", justify="right")
        for comp in vr.comparisons:
            delta_color = "green" if comp.delta_pct < 0 else "red" if comp.delta_pct > 0 else ""
            delta_str = f"{comp.delta_pct:+.1f}%"
            if delta_color:
                delta_str = f"[{delta_color}]{delta_str}[/{delta_color}]"
            table.add_row(
                comp.target,
                f"{comp.full_rmse:.4f}",
                f"{comp.pruned_rmse:.4f}",
                delta_str,
            )
        console.print(table)


def print_routing_table(routes: dict[str, str]) -> None:
    """Print the effective stat→system routing table."""
    table = Table(title="Routing Table", show_edge=False)
    table.add_column("Stat", style="bold")
    table.add_column("System")
    for stat in sorted(routes):
        table.add_row(stat, routes[stat])
    console.print(table)


def print_coverage_matrix(
    system_stats: dict[str, set[str]],
    routes: dict[str, str] | None = None,
    required_stats: frozenset[str] | None = None,
) -> None:
    """Print a stat × system coverage matrix."""
    systems = sorted(system_stats)
    all_stats = sorted({s for stats in system_stats.values() for s in stats})

    table = Table(title="Coverage Matrix", show_edge=False)
    table.add_column("Stat", style="bold")
    for sys_name in systems:
        table.add_column(sys_name, justify="center")
    if routes:
        table.add_column("Routed To", style="cyan")
    if required_stats:
        table.add_column("Required", justify="center")

    for stat in all_stats:
        row: list[str] = []
        for sys_name in systems:
            row.append("[green]✓[/green]" if stat in system_stats[sys_name] else "[dim]·[/dim]")
        if routes:
            row.append(routes.get(stat, "[dim]—[/dim]"))
        if required_stats:
            if stat in required_stats:
                covered = routes is None or stat in routes
                row.append("[green]✓[/green]" if covered else "[red]✗[/red]")
            else:
                row.append("[dim]·[/dim]")

        stat_style = ""
        if required_stats and stat in required_stats and routes and stat not in routes:
            stat_style = "red bold"
        table.add_row(f"[{stat_style}]{stat}[/{stat_style}]" if stat_style else stat, *row)
    console.print(table)


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

    # Per-target optimal breakdown
    for label, per_target_best in [
        ("Batter", result.batter_per_target_best),
        ("Pitcher", result.pitcher_per_target_best),
    ]:
        if not per_target_best:
            continue
        has_divergence = any(ptb.delta_pct > 0.0 for ptb in per_target_best.values())
        if not has_divergence:
            continue
        console.print(f"[bold]{label} per-target optimal params:[/bold]")
        for target_name, ptb in sorted(per_target_best.items()):
            if ptb.delta_pct > 0.0:
                params_str = ", ".join(f"{k}={v!r}" for k, v in sorted(ptb.best_params.items()))
                console.print(f"  {target_name}: RMSE {ptb.best_rmse:.4f} (joint +{ptb.delta_pct:.1f}%) — {params_str}")
            else:
                console.print(f"  {target_name}: RMSE {ptb.best_rmse:.4f} (joint is optimal)")
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
