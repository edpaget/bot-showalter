from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from fantasy_baseball_manager.draft.models import DraftRanking
    from fantasy_baseball_manager.draft.simulation_models import (
        SimulationResult,
        TeamResult,
    )

from fantasy_baseball_manager.valuation.models import StatCategory

console = Console()

# Categories where lower is better (invert ranking)
_LOWER_IS_BETTER: frozenset[StatCategory] = frozenset({StatCategory.ERA, StatCategory.WHIP})


def print_pick_log(result: SimulationResult) -> None:
    """Print pick-by-pick draft log."""
    table = Table(title="Draft Pick Log")
    table.add_column("Pick", justify="right")
    table.add_column("Rd", justify="right")
    table.add_column("Team")
    table.add_column("Player")
    table.add_column("Pos")
    table.add_column("Value", justify="right")

    for pick in result.pick_log:
        if pick.overall_pick == 0:
            continue  # Skip keepers in log (round 0)
        table.add_row(
            str(pick.overall_pick),
            str(pick.round_number),
            pick.team_name,
            pick.player_name,
            pick.position or "-",
            f"{pick.adjusted_value:.1f}",
        )
    console.print(table)


def print_team_roster(team_result: TeamResult) -> None:
    """Print a single team's roster."""
    table = Table(title=f"{team_result.team_name} ({team_result.strategy_name})")
    table.add_column("Rd", justify="right")
    table.add_column("Pick", justify="right")
    table.add_column("Player")
    table.add_column("Pos")
    table.add_column("Value", justify="right")

    for pick in team_result.picks:
        rd = str(pick.round_number) if pick.round_number > 0 else "K"
        pk = str(pick.overall_pick) if pick.overall_pick > 0 else "-"
        table.add_row(
            rd,
            pk,
            pick.player_name,
            pick.position or "-",
            f"{pick.adjusted_value:.1f}",
        )
    console.print(table)

    # Category totals
    if team_result.category_totals:
        cat_parts: list[str] = []
        for cat, total in sorted(team_result.category_totals.items(), key=lambda x: x[0].value):
            if cat in _LOWER_IS_BETTER:
                cat_parts.append(f"{cat.value}: {total:.2f}")
            else:
                cat_parts.append(f"{cat.value}: {total:.0f}")
        console.print("  ".join(cat_parts))


def print_standings(result: SimulationResult) -> None:
    """Print projected roto standings."""
    # Collect all categories across teams
    all_categories: set[StatCategory] = set()
    for tr in result.team_results:
        all_categories.update(tr.category_totals.keys())

    categories = sorted(all_categories, key=lambda c: c.value)
    if not categories:
        console.print("No category data available.")
        return

    num_teams = len(result.team_results)

    # Rank teams per category
    category_points: dict[int, dict[StatCategory, int]] = defaultdict(dict)
    for cat in categories:
        team_vals: list[tuple[int, float]] = []
        for tr in result.team_results:
            team_vals.append((tr.team_id, tr.category_totals.get(cat, 0.0)))

        # Sort: lower is better for ERA/WHIP, higher is better for rest
        reverse = cat not in _LOWER_IS_BETTER
        team_vals.sort(key=lambda x: x[1], reverse=reverse)

        for rank, (team_id, _) in enumerate(team_vals):
            points = num_teams - rank  # 1st place gets num_teams pts
            category_points[team_id][cat] = points

    # Build table
    table = Table(title="Projected Standings")
    table.add_column("Team")
    for cat in categories:
        table.add_column(cat.value, justify="right")
    table.add_column("Total", justify="right")

    # Sort by total points descending
    team_totals: list[tuple[TeamResult, int]] = []
    for tr in result.team_results:
        total = sum(category_points[tr.team_id].values())
        team_totals.append((tr, total))
    team_totals.sort(key=lambda x: x[1], reverse=True)

    for tr, total in team_totals:
        pts = category_points[tr.team_id]
        row = [tr.team_name]
        for cat in categories:
            row.append(str(pts.get(cat, 0)))
        row.append(str(total))
        table.add_row(*row)

    console.print(table)


def print_draft_rankings(rankings: list[DraftRanking], year: int) -> None:
    """Print a draft rankings table."""
    console.print(f"[bold]Draft rankings for {year}:[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Rk", justify="right")
    table.add_column("Name")
    table.add_column("Pos")
    table.add_column("Mult", justify="right")
    table.add_column("Raw", justify="right")
    table.add_column("Wtd", justify="right")
    table.add_column("Adj", justify="right")

    for r in rankings:
        display_pos = tuple(p for p in r.eligible_positions if p != "Util") or r.eligible_positions
        pos_str = "/".join(display_pos) if display_pos else "-"
        table.add_row(
            str(r.rank),
            r.name,
            pos_str,
            f"{r.position_multiplier:.2f}",
            f"{r.raw_value:.1f}",
            f"{r.weighted_value:.1f}",
            f"{r.adjusted_value:.1f}",
        )

    console.print(table)


def print_full_report(result: SimulationResult) -> None:
    """Print a complete draft simulation report."""
    # Pick log
    print_pick_log(result)

    # Team rosters
    console.print("\n[bold]Team Rosters[/bold]")
    for tr in result.team_results:
        print_team_roster(tr)
        console.print()

    # Standings
    print_standings(result)
