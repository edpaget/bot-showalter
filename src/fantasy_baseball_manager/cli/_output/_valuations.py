from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        PlayerValuation,
        ValuationComparisonResult,
        ValuationEvalResult,
        ValuationRegressionCheck,
    )


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


def _print_metrics_block(result: ValuationEvalResult) -> None:
    """Print the MAE / ρ / WAR / hit-rate lines for a single result."""
    console.print(f"  Value MAE: [bold]{result.value_mae:.2f}[/bold]")
    console.print(f"  Spearman rank correlation: [bold]{result.rank_correlation:.4f}[/bold]")

    if result.war_correlation is not None:
        console.print(f"  WAR correlation (all):      [bold]{result.war_correlation:.4f}[/bold]")
    if result.war_correlation_batters is not None:
        console.print(f"  WAR correlation (batters):  [bold]{result.war_correlation_batters:.4f}[/bold]")
    if result.war_correlation_pitchers is not None:
        console.print(f"  WAR correlation (pitchers): [bold]{result.war_correlation_pitchers:.4f}[/bold]")
    if result.war_correlation_sp is not None:
        console.print(f"  WAR correlation (SP):       [bold]{result.war_correlation_sp:.4f}[/bold]")

    if result.hit_rates:
        parts = [f"top-{n}: {rate:.1f}%" for n, rate in sorted(result.hit_rates.items())]
        console.print(f"  Top-N hit rate:  {'  '.join(parts)}")

    if result.category_hit_rates:
        cat_parts = [f"{cat}: {rate:.1f}%" for cat, rate in sorted(result.category_hit_rates.items())]
        console.print(f"  Category hit rate (top-20):  {'  '.join(cat_parts)}")


def print_valuation_eval_result(result: ValuationEvalResult, top: int | None = None) -> None:
    """Print valuation evaluation results with metrics and per-player breakdown."""
    if result.n == 0:
        console.print("No matched players found.")
        return

    if result.filter_description is not None and result.total_matched is not None:
        population_info = f"{result.n} of {result.total_matched} matched players, {result.filter_description}"
    else:
        population_info = f"{result.n} matched players"
    console.print(
        f"Valuation evaluation: [bold]{result.system}[/bold] v{result.version}"
        f" — season {result.season} ({population_info})"
    )
    _print_metrics_block(result)

    # Cohort output (stratification)
    if result.cohorts is not None:
        for label, cohort in sorted(result.cohorts.items()):
            console.print()
            console.print(f"  [bold]--- {label.title()}s (n={cohort.n}) ---[/bold]")
            _print_metrics_block(cohort)

    # Tail output
    if result.tail_results is not None:
        for n, tail in sorted(result.tail_results.items()):
            console.print()
            console.print(f"  [bold]--- Top {n} ---[/bold]")
            _print_metrics_block(tail)

    console.print()

    players = result.players
    if top is not None:
        players = players[:top]

    has_war = any(p.actual_war is not None for p in players)

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Player")
    table.add_column("Type")
    table.add_column("Predicted$", justify="right")
    table.add_column("Actual$", justify="right")
    table.add_column("Surplus$", justify="right")
    table.add_column("PredRank", justify="right")
    table.add_column("ActRank", justify="right")
    if has_war:
        table.add_column("WAR", justify="right")
    for p in players:
        surplus_str = f"{p.surplus:+.1f}"
        row = [
            p.player_name,
            p.player_type,
            f"${p.predicted_value:.1f}",
            f"${p.actual_value:.1f}",
            surplus_str,
            str(p.predicted_rank),
            str(p.actual_rank),
        ]
        if has_war:
            row.append(f"{p.actual_war:.1f}" if p.actual_war is not None else "—")
        table.add_row(*row)
    console.print(table)


def _fmt_delta(candidate: float, baseline: float, fmt: str = ".2f", suffix: str = "") -> str:
    """Format a delta value with +/- prefix."""
    delta = candidate - baseline
    return f"{delta:+{fmt}}{suffix}"


def print_valuation_comparison(result: ValuationComparisonResult) -> None:
    """Print side-by-side comparison of two valuation evaluation results."""
    b = result.baseline
    c = result.candidate

    b_label = f"{b.system}/{b.version}"
    c_label = f"{c.system}/{c.version}"

    console.print(f"Valuation comparison — season {result.season}")
    console.print()

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("Metric")
    table.add_column(f"baseline ({b_label})", justify="right")
    table.add_column(f"candidate ({c_label})", justify="right")
    table.add_column("Δ", justify="right")

    # Value MAE (lower is better)
    table.add_row("Value MAE", f"{b.value_mae:.2f}", f"{c.value_mae:.2f}", _fmt_delta(c.value_mae, b.value_mae))

    # Rank ρ
    table.add_row(
        "Rank ρ",
        f"{b.rank_correlation:.4f}",
        f"{c.rank_correlation:.4f}",
        _fmt_delta(c.rank_correlation, b.rank_correlation, ".4f"),
    )

    # WAR correlations
    if b.war_correlation is not None and c.war_correlation is not None:
        table.add_row(
            "WAR ρ (all)",
            f"{b.war_correlation:.4f}",
            f"{c.war_correlation:.4f}",
            _fmt_delta(c.war_correlation, b.war_correlation, ".4f"),
        )
    if b.war_correlation_batters is not None and c.war_correlation_batters is not None:
        table.add_row(
            "WAR ρ (batters)",
            f"{b.war_correlation_batters:.4f}",
            f"{c.war_correlation_batters:.4f}",
            _fmt_delta(c.war_correlation_batters, b.war_correlation_batters, ".4f"),
        )
    if b.war_correlation_pitchers is not None and c.war_correlation_pitchers is not None:
        table.add_row(
            "WAR ρ (pitchers)",
            f"{b.war_correlation_pitchers:.4f}",
            f"{c.war_correlation_pitchers:.4f}",
            _fmt_delta(c.war_correlation_pitchers, b.war_correlation_pitchers, ".4f"),
        )
    if b.war_correlation_sp is not None and c.war_correlation_sp is not None:
        table.add_row(
            "WAR ρ (SP)",
            f"{b.war_correlation_sp:.4f}",
            f"{c.war_correlation_sp:.4f}",
            _fmt_delta(c.war_correlation_sp, b.war_correlation_sp, ".4f"),
        )

    # Hit rates
    if b.hit_rates and c.hit_rates:
        common_ns = sorted(set(b.hit_rates) & set(c.hit_rates))
        for n in common_ns:
            table.add_row(
                f"Hit rate top-{n}",
                f"{b.hit_rates[n]:.1f}%",
                f"{c.hit_rates[n]:.1f}%",
                _fmt_delta(c.hit_rates[n], b.hit_rates[n], ".1f", "pp"),
            )

    # Category hit rates
    if b.category_hit_rates and c.category_hit_rates:
        common_cats = sorted(set(b.category_hit_rates) & set(c.category_hit_rates))
        for cat in common_cats:
            table.add_row(
                f"Cat hit rate ({cat})",
                f"{b.category_hit_rates[cat]:.1f}%",
                f"{c.category_hit_rates[cat]:.1f}%",
                _fmt_delta(c.category_hit_rates[cat], b.category_hit_rates[cat], ".1f", "pp"),
            )

    console.print(table)
    console.print(f"\n  Population: baseline n={b.n}, candidate n={c.n}")


def print_valuation_regression_check(check: ValuationRegressionCheck) -> None:
    """Print the regression check result."""
    style = "bold green" if check.passed else "bold red"
    console.print(f"\nRegression check: [{style}]{check.explanation}[/{style}]")
