from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import PlayerValuation, ValuationEvalResult


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
