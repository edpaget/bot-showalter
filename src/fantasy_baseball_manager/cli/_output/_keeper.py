from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        AdjustedValuation,
        KeeperDecision,
        KeeperScenario,
        KeeperSolution,
        KeeperTradeImpact,
        TradeEvaluation,
        TradePlayerDetail,
    )


def print_keeper_decisions(decisions: list[KeeperDecision]) -> None:
    table = Table(title="Keeper Decisions", show_edge=False, pad_edge=False)
    table.add_column("Player", justify="left")
    table.add_column("Pos", justify="left")
    table.add_column("Cost", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Surplus", justify="right")
    table.add_column("Yrs", justify="right")
    table.add_column("Rec", justify="left")

    for d in decisions:
        style = "green" if d.surplus >= 0 else "red"
        table.add_row(
            d.player_name,
            d.position,
            f"${d.cost:.0f}",
            f"${d.projected_value:.0f}",
            f"${d.surplus:.1f}",
            str(d.years_remaining),
            d.recommendation,
            style=style,
        )

    console.print(table)


def print_adjusted_rankings(rankings: list[AdjustedValuation], *, top: int | None = None) -> None:
    table = Table(title="Keeper-Adjusted Rankings", show_edge=False, pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Player", justify="left")
    table.add_column("Type", justify="left")
    table.add_column("Pos", justify="left")
    table.add_column("Original", justify="right")
    table.add_column("Adjusted", justify="right")
    table.add_column("Change", justify="right")

    display = rankings[:top] if top is not None else rankings
    for i, r in enumerate(display, 1):
        if abs(r.value_change) > 3:
            style = "bold green" if r.value_change > 0 else "bold red"
        elif r.value_change > 0:
            style = "green"
        elif r.value_change < 0:
            style = "red"
        else:
            style = ""
        sign = "+" if r.value_change > 0 else ""
        table.add_row(
            str(i),
            r.player_name,
            r.player_type,
            r.position,
            f"${r.original_value:.1f}",
            f"${r.adjusted_value:.1f}",
            f"{sign}${r.value_change:.1f}",
            style=style,
        )

    console.print(table)


def _trade_player_table(label: str, details: list[TradePlayerDetail]) -> Table:
    table = Table(title=label, show_edge=False, pad_edge=False)
    table.add_column("Player", justify="left")
    table.add_column("Pos", justify="left")
    table.add_column("Cost", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Surplus", justify="right")
    table.add_column("Yrs", justify="right")

    for d in details:
        style = "green" if d.surplus >= 0 else "red"
        table.add_row(
            d.player_name,
            d.position,
            f"${d.cost:.0f}",
            f"${d.projected_value:.1f}",
            f"${d.surplus:.1f}",
            str(d.years_remaining),
            style=style,
        )
    return table


def print_trade_evaluation(evaluation: TradeEvaluation) -> None:
    console.print()
    console.print(_trade_player_table("You give", evaluation.team_a_gives))
    console.print()
    console.print(_trade_player_table("You receive", evaluation.team_b_gives))
    console.print()

    a_sign = "+" if evaluation.team_a_surplus_delta >= 0 else ""
    b_sign = "+" if evaluation.team_b_surplus_delta >= 0 else ""
    a_style = "green" if evaluation.team_a_surplus_delta >= 0 else "red"
    b_style = "green" if evaluation.team_b_surplus_delta >= 0 else "red"

    console.print(f"Your surplus delta: [{a_style}]{a_sign}${evaluation.team_a_surplus_delta:.1f}[/{a_style}]")
    console.print(f"Their surplus delta: [{b_style}]{b_sign}${evaluation.team_b_surplus_delta:.1f}[/{b_style}]")

    if evaluation.winner == "team_a":
        console.print("[bold green]Verdict: You win this trade[/bold green]")
    elif evaluation.winner == "team_b":
        console.print("[bold red]Verdict: They win this trade[/bold red]")
    else:
        console.print("[bold]Verdict: Even trade[/bold]")


def print_keeper_solution(solution: KeeperSolution) -> None:
    """Print the optimal keeper set with alternatives and sensitivity analysis."""
    optimal = solution.optimal

    # Optimal keeper set table
    table = Table(title="Optimal Keeper Set", show_edge=False, pad_edge=False)
    table.add_column("Player", justify="left")
    table.add_column("Pos", justify="left")
    table.add_column("Cost", justify="right")
    table.add_column("Value", justify="right")
    table.add_column("Surplus", justify="right")

    for p in optimal.players:
        table.add_row(
            p.player_name,
            p.position,
            f"${p.cost:.0f}",
            f"${p.projected_value:.0f}",
            f"${p.surplus:.1f}",
        )

    console.print(table)
    console.print(
        f"Total surplus: ${optimal.total_surplus:.1f}  "
        f"Total cost: ${optimal.total_cost:.0f}  "
        f"Score: {optimal.score:.1f}"
    )
    console.print()

    # Alternatives
    if solution.alternatives:
        console.print("[bold]Alternatives[/bold]")
        for i, alt in enumerate(solution.alternatives[:3], 1):
            names = ", ".join(p.player_name for p in alt.players)
            gap = optimal.score - alt.score
            console.print(f"  Alt {i}: {names}  (score: {alt.score:.1f}, gap: -{gap:.1f})")
        console.print()

    # Sensitivity
    if solution.sensitivity:
        sens_table = Table(title="Sensitivity", show_edge=False, pad_edge=False)
        sens_table.add_column("Player", justify="left")
        sens_table.add_column("Surplus Gap", justify="right")

        for entry in solution.sensitivity:
            gap_str = f"{entry.surplus_gap:.1f}" if entry.surplus_gap != float("inf") else "inf"
            sens_table.add_row(entry.player_name, gap_str)

        console.print(sens_table)


def print_keeper_scenarios(scenarios: list[KeeperScenario]) -> None:
    """Print scenario comparison ranked by score."""
    table = Table(title="Scenario Comparison", show_edge=False, pad_edge=False)
    table.add_column("Rank", justify="right")
    table.add_column("Name", justify="left")
    table.add_column("Score", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Keepers", justify="left")

    for i, s in enumerate(scenarios, 1):
        names = ", ".join(p.player_name for p in s.keeper_set.players)
        delta_str = "-" if s.delta_vs_optimal == 0.0 else f"+${s.delta_vs_optimal:.1f}"
        style = "green" if i == 1 else ""
        table.add_row(
            str(i),
            s.name,
            f"{s.keeper_set.score:.1f}",
            delta_str,
            names,
            style=style,
        )

    console.print(table)


def print_keeper_trade_impact(impact: KeeperTradeImpact) -> None:
    """Print before/after optimal sets and score delta for a trade."""
    before_names = ", ".join(p.player_name for p in impact.before.optimal.players)
    after_names = ", ".join(p.player_name for p in impact.after.optimal.players)

    console.print("[bold]Before[/bold]")
    console.print(f"  Keepers: {before_names}")
    console.print(f"  Score: {impact.before.optimal.score:.1f}")
    console.print()
    console.print("[bold]After[/bold]")
    console.print(f"  Keepers: {after_names}")
    console.print(f"  Score: {impact.after.optimal.score:.1f}")
    console.print()

    color = "green" if impact.score_delta >= 0 else "red"
    delta_str = f"+${impact.score_delta:.1f}" if impact.score_delta >= 0 else f"-${abs(impact.score_delta):.1f}"
    console.print(f"Score delta: [{color}]{delta_str}[/{color}]")
