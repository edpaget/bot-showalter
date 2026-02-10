from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from fantasy_baseball_manager.adp.name_utils import normalize_name

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fantasy_baseball_manager.adp.models import ADPEntry
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


_PITCHER_POSITIONS: frozenset[str] = frozenset({"SP", "RP", "P"})


def _parse_player_name(name: str, positions: tuple[str, ...]) -> tuple[str, str | None]:
    """Parse player name and determine position type.

    Handles multiple ADP source formats:
    - Yahoo: Uses (Batter)/(Pitcher) suffixes for two-way players
    - ESPN: Uses positions like "DH, SP" for two-way players

    Args:
        name: Player name from ADP source.
        positions: Player's positions from the ADP entry.

    Returns:
        Tuple of (normalized_name, position_type) where position_type is
        "B" for batters, "P" for pitchers, or None for non-two-way players.
    """
    # Check for Yahoo-style suffixes first
    match = re.search(r"\s*\((Batter|Pitcher)\)\s*$", name)
    if match:
        suffix = match.group(1)
        position_type: str | None = "B" if suffix == "Batter" else "P"
    else:
        # For sources without suffixes (like ESPN), there's only one entry
        # for two-way players. Store them as untyped so they match any lookup.
        position_type = None

    return normalize_name(name), position_type


class _ADPLookup:
    """ADP lookup that handles two-way players correctly.

    Supports multiple ADP source formats:
    - Yahoo: Uses (Batter)/(Pitcher) suffixes for two-way players
    - ESPN: Uses positions to identify player type
    - Composite: Handles averaged data from multiple sources
    """

    def __init__(self, entries: Sequence[ADPEntry]) -> None:
        # For two-way players: (name, position_type) -> ADP
        self._typed: dict[tuple[str, str], float] = {}
        # For regular players: name -> ADP
        self._untyped: dict[str, float] = {}

        for entry in entries:
            normalized, position_type = _parse_player_name(entry.name, entry.positions)
            if position_type is not None:
                self._typed[(normalized, position_type)] = entry.adp
            else:
                self._untyped[normalized] = entry.adp

    def get(self, name: str, is_pitcher: bool) -> float | None:
        """Look up ADP for a player.

        Args:
            name: Player name from projection.
            is_pitcher: True if this is a pitcher projection.

        Returns:
            ADP value or None if not found.
        """
        normalized = normalize_name(name)
        position_type = "P" if is_pitcher else "B"

        # First try position-specific lookup (for two-way players)
        adp = self._typed.get((normalized, position_type))
        if adp is not None:
            return adp

        # Fall back to untyped lookup
        return self._untyped.get(normalized)


def print_draft_rankings(
    rankings: list[DraftRanking], year: int, adp_entries: Sequence[ADPEntry] | None = None
) -> None:
    """Print a draft rankings table.

    Args:
        rankings: List of draft rankings to display.
        year: The season year.
        adp_entries: Optional ADP entries for comparison. If provided, adds ADP and Diff columns.
    """
    console.print(f"[bold]Draft rankings for {year}:[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Rk", justify="right")
    table.add_column("Name")
    table.add_column("Pos")
    table.add_column("Mult", justify="right")
    table.add_column("Raw", justify="right")
    table.add_column("Wtd", justify="right")
    table.add_column("Adj", justify="right")

    if adp_entries is not None:
        table.add_column("ADP", justify="right")
        table.add_column("Diff", justify="right")
        adp_lookup = _ADPLookup(adp_entries)
    else:
        adp_lookup = None

    for r in rankings:
        display_pos = tuple(p for p in r.eligible_positions if p != "Util") or r.eligible_positions
        pos_str = "/".join(display_pos) if display_pos else "-"
        row = [
            str(r.rank),
            r.name,
            pos_str,
            f"{r.position_multiplier:.2f}",
            f"{r.raw_value:.1f}",
            f"{r.weighted_value:.1f}",
            f"{r.adjusted_value:.1f}",
        ]

        if adp_lookup is not None:
            # Determine if this is a pitcher based on eligible positions
            is_pitcher = bool(set(r.eligible_positions) & _PITCHER_POSITIONS)
            adp = adp_lookup.get(r.name, is_pitcher)
            if adp is not None:
                diff = round(adp - r.rank)
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                row.extend([f"{adp:.1f}", diff_str])
            else:
                row.extend(["-", "-"])

        table.add_row(*row)

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
