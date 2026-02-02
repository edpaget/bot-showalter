from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.draft.simulation_models import (
        SimulationPick,
        SimulationResult,
        TeamResult,
    )

from fantasy_baseball_manager.valuation.models import StatCategory

# Categories where lower is better (invert ranking)
_LOWER_IS_BETTER: frozenset[StatCategory] = frozenset({StatCategory.ERA, StatCategory.WHIP})


def format_pick_log(result: SimulationResult) -> str:
    """Format pick-by-pick draft log."""
    lines: list[str] = []
    lines.append("Draft Pick Log")
    lines.append("=" * 80)
    header = f"{'Pick':>4} {'Rd':>3} {'Team':<20} {'Player':<25} {'Pos':<5} {'Value':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for pick in result.pick_log:
        if pick.overall_pick == 0:
            continue  # Skip keepers in log (round 0)
        lines.append(
            f"{pick.overall_pick:>4} {pick.round_number:>3} {pick.team_name:<20}"
            f" {pick.player_name:<25} {pick.position or '-':<5} {pick.adjusted_value:>7.1f}"
        )
    return "\n".join(lines)


def format_team_roster(team_result: TeamResult, pick_log: tuple[SimulationPick, ...]) -> str:
    """Format a single team's roster."""
    lines: list[str] = []
    lines.append(f"\n{team_result.team_name} ({team_result.strategy_name})")
    lines.append("-" * 60)
    header = f"{'Rd':>3} {'Pick':>4} {'Player':<25} {'Pos':<5} {'Value':>7}"
    lines.append(header)

    for pick in team_result.picks:
        rd = pick.round_number if pick.round_number > 0 else "K"
        pk = pick.overall_pick if pick.overall_pick > 0 else "-"
        lines.append(
            f"{rd!s:>3} {pk!s:>4} {pick.player_name:<25} {pick.position or '-':<5}" f" {pick.adjusted_value:>7.1f}"
        )

    # Category totals
    if team_result.category_totals:
        lines.append("")
        cat_parts: list[str] = []
        for cat, total in sorted(team_result.category_totals.items(), key=lambda x: x[0].value):
            if cat in _LOWER_IS_BETTER:
                cat_parts.append(f"{cat.value}: {total:.2f}")
            else:
                cat_parts.append(f"{cat.value}: {total:.0f}")
        lines.append("  ".join(cat_parts))

    return "\n".join(lines)


def format_standings(result: SimulationResult) -> str:
    """Format projected roto standings."""
    lines: list[str] = []
    lines.append("\nProjected Standings")
    lines.append("=" * 80)

    # Collect all categories across teams
    all_categories: set[StatCategory] = set()
    for tr in result.team_results:
        all_categories.update(tr.category_totals.keys())

    categories = sorted(all_categories, key=lambda c: c.value)
    if not categories:
        lines.append("No category data available.")
        return "\n".join(lines)

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
    cat_headers = "".join(f"{c.value:>6}" for c in categories)
    header = f"{'Team':<20}{cat_headers}{'Total':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    # Sort by total points descending
    team_totals: list[tuple[TeamResult, int]] = []
    for tr in result.team_results:
        total = sum(category_points[tr.team_id].values())
        team_totals.append((tr, total))
    team_totals.sort(key=lambda x: x[1], reverse=True)

    for tr, total in team_totals:
        pts = category_points[tr.team_id]
        cat_vals = "".join(f"{pts.get(c, 0):>6}" for c in categories)
        lines.append(f"{tr.team_name:<20}{cat_vals}{total:>7}")

    return "\n".join(lines)


def format_full_report(result: SimulationResult) -> str:
    """Format a complete draft simulation report."""
    sections: list[str] = []

    # Pick log
    sections.append(format_pick_log(result))

    # Team rosters
    sections.append("\n\nTeam Rosters")
    sections.append("=" * 80)
    for tr in result.team_results:
        sections.append(format_team_roster(tr, result.pick_log))

    # Standings
    sections.append(format_standings(result))

    return "\n".join(sections)
