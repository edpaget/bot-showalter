from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import (
        AvailabilityWindow,
        BatchSimulationResult,
        DraftResult,
        PlayerAvailabilityCurve,
        StrategyComparison,
    )


def print_mock_draft_result(result: DraftResult, user_team: int) -> None:
    """Print draft log and user roster summary."""
    # Draft Log table
    table = Table(title="Draft Log", show_edge=False, pad_edge=False)
    table.add_column("Pick", justify="right")
    table.add_column("Rd", justify="right")
    table.add_column("Team", justify="right")
    table.add_column("Player")
    table.add_column("Pos")
    table.add_column("Value", justify="right")

    for pick in result.picks:
        style = "green" if pick.team_idx == user_team else ""
        table.add_row(
            str(pick.pick),
            str(pick.round),
            str(pick.team_idx),
            pick.player_name,
            pick.position,
            f"${pick.value:.1f}",
            style=style,
        )

    console.print(table)
    console.print()

    # User Roster table
    roster = result.rosters.get(user_team, [])
    roster_table = Table(title="Your Roster", show_edge=False, pad_edge=False)
    roster_table.add_column("Rd", justify="right")
    roster_table.add_column("Player")
    roster_table.add_column("Pos")
    roster_table.add_column("Value", justify="right")

    total_value = 0.0
    for pick in roster:
        total_value += pick.value
        roster_table.add_row(
            str(pick.round),
            pick.player_name,
            pick.position,
            f"${pick.value:.1f}",
        )

    console.print(roster_table)
    console.print(f"Total value: ${total_value:.1f}")


def print_batch_simulation_result(result: BatchSimulationResult, top_n: int = 20) -> None:
    """Print batch simulation summary, top players, and strategy comparison."""
    s = result.summary

    # Simulation Summary
    summary_table = Table(title="Simulation Summary", show_edge=False, pad_edge=False)
    summary_table.add_column("Metric")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Simulations", str(s.n_simulations))
    if s.team_idx is not None:
        summary_table.add_row("Position", str(s.team_idx + 1))
    summary_table.add_row("Avg", f"${s.avg_roster_value:.1f}")
    summary_table.add_row("Median", f"${s.median_roster_value:.1f}")
    summary_table.add_row("P10", f"${s.p10_roster_value:.1f}")
    summary_table.add_row("P25", f"${s.p25_roster_value:.1f}")
    summary_table.add_row("P75", f"${s.p75_roster_value:.1f}")
    summary_table.add_row("P90", f"${s.p90_roster_value:.1f}")

    console.print(summary_table)
    console.print()

    # Top Drafted Players
    drafted = [f for f in result.player_frequencies if f.pct_drafted > 0]
    drafted.sort(key=lambda f: f.pct_drafted, reverse=True)
    top = drafted[:top_n]

    if top:
        player_table = Table(title="Top Drafted Players", show_edge=False, pad_edge=False)
        player_table.add_column("Player")
        player_table.add_column("%", justify="right")
        player_table.add_column("Avg Rd", justify="right")
        player_table.add_column("Avg Pick", justify="right")

        for f in top:
            player_table.add_row(
                f.player_name,
                f"{f.pct_drafted:.0%}",
                f"{f.avg_round_drafted:.1f}",
                f"{f.avg_pick_drafted:.1f}",
            )
        console.print(player_table)
        console.print()

    # Strategy Comparison
    if result.strategy_comparisons:
        print_strategy_comparison_table(result.strategy_comparisons)


def print_strategy_comparison_table(comparisons: list[StrategyComparison]) -> None:
    """Print strategy comparison table sorted by win rate descending."""
    sorted_comps = sorted(comparisons, key=lambda c: c.win_rate, reverse=True)

    table = Table(title="Strategy Comparison", show_edge=False, pad_edge=False)
    table.add_column("Strategy")
    table.add_column("Avg Value", justify="right")
    table.add_column("Win Rate", justify="right")

    best_rate = sorted_comps[0].win_rate if sorted_comps else 0.0
    for comp in sorted_comps:
        style = "green" if comp.win_rate == best_rate else ""
        table.add_row(
            comp.strategy_name,
            f"${comp.avg_value:.1f}",
            f"{comp.win_rate:.1%}",
            style=style,
        )

    console.print(table)


def print_availability_windows(windows: list[AvailabilityWindow]) -> None:
    """Print player availability windows as a Rich table."""
    table = Table(title="Player Availability Windows", show_edge=False, pad_edge=False)
    table.add_column("Player")
    table.add_column("Pos")
    table.add_column("Earliest", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Latest", justify="right")
    table.add_column("Avail%", justify="right")

    for w in windows:
        avail_style = "green" if w.available_at_user_pick >= 0.8 else ("red" if w.available_at_user_pick < 0.3 else "")
        table.add_row(
            w.player_name,
            w.position,
            f"{w.earliest_pick:.0f}",
            f"{w.median_pick:.0f}",
            f"{w.latest_pick:.0f}",
            f"{w.available_at_user_pick:.0%}",
            style=avail_style,
        )

    console.print(table)


def print_player_availability_curve(curve: PlayerAvailabilityCurve) -> None:
    """Print round-by-round availability curve for a single player."""
    console.print(f"\n[bold]{curve.player_name}[/bold] ({curve.position})")
    table = Table(title="Availability Curve", show_edge=False, pad_edge=False)
    table.add_column("Round", justify="right")
    table.add_column("Pick#", justify="right")
    table.add_column("Available%", justify="right")

    for pa in curve.pick_availabilities:
        style = "green" if pa.probability >= 0.8 else ("red" if pa.probability < 0.3 else "")
        table.add_row(
            str(pa.round),
            str(pa.pick),
            f"{pa.probability:.0%}",
            style=style,
        )

    console.print(table)
