"""Formatting functions for agent tool output.

This module contains text table formatting for player valuations,
comparisons, and keeper rankings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fantasy_baseball_manager.valuation.models import PlayerValue


def format_batter_table(values: list[PlayerValue], top_n: int) -> str:
    """Format a list of batter valuations as a text table."""
    sorted_values = sorted(values, key=lambda p: p.total_value, reverse=True)[:top_n]

    lines: list[str] = []
    lines.append(f"Top {len(sorted_values)} Projected Batters:")
    lines.append("")
    header = f"{'Rk':>4} {'Name':<25} {'Value':>7}"
    # Add category columns
    if sorted_values and sorted_values[0].category_values:
        for cv in sorted_values[0].category_values:
            header += f" {cv.category.value:>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, pv in enumerate(sorted_values, start=1):
        row = f"{i:>4} {pv.name:<25} {pv.total_value:>7.1f}"
        for cv in pv.category_values:
            row += f" {cv.raw_stat:>6.1f}"
        lines.append(row)

    return "\n".join(lines)


def format_pitcher_table(values: list[PlayerValue], top_n: int) -> str:
    """Format a list of pitcher valuations as a text table."""
    sorted_values = sorted(values, key=lambda p: p.total_value, reverse=True)[:top_n]

    lines: list[str] = []
    lines.append(f"Top {len(sorted_values)} Projected Pitchers:")
    lines.append("")
    header = f"{'Rk':>4} {'Name':<25} {'Value':>7}"
    if sorted_values and sorted_values[0].category_values:
        for cv in sorted_values[0].category_values:
            header += f" {cv.category.value:>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, pv in enumerate(sorted_values, start=1):
        row = f"{i:>4} {pv.name:<25} {pv.total_value:>7.1f}"
        for cv in pv.category_values:
            # Format ERA/WHIP with more decimals
            if cv.category.value in ("ERA", "WHIP"):
                row += f" {cv.raw_stat:>6.2f}"
            else:
                row += f" {cv.raw_stat:>6.1f}"
        lines.append(row)

    return "\n".join(lines)


def format_player_lookup(matches: list[PlayerValue]) -> str:
    """Format player lookup results with detailed stats."""
    lines: list[str] = []
    for pv in sorted(matches, key=lambda p: p.total_value, reverse=True):
        pos_type = "Batter" if pv.position_type == "B" else "Pitcher"
        lines.append(f"{pv.name} ({pos_type})")
        lines.append(f"  Total Z-Score Value: {pv.total_value:.2f}")
        lines.append("  Category Breakdown:")
        for cv in pv.category_values:
            if cv.category.value in ("ERA", "WHIP"):
                lines.append(f"    {cv.category.value}: {cv.raw_stat:.2f} (z={cv.value:.2f})")
            else:
                lines.append(f"    {cv.category.value}: {cv.raw_stat:.1f} (z={cv.value:.2f})")
        lines.append("")

    return "\n".join(lines)


def format_player_comparison(
    found_players: list[PlayerValue],
    not_found: list[str],
) -> str:
    """Format a side-by-side player comparison table."""
    lines: list[str] = []

    if not_found:
        lines.append(f"Note: Could not find: {', '.join(not_found)}")
        lines.append("")

    # Build comparison table
    lines.append("Player Comparison:")
    lines.append("")

    # Header row
    header = f"{'Stat':<12}"
    for pv in found_players:
        header += f" {pv.name[:15]:<15}"
    lines.append(header)
    lines.append("-" * len(header))

    # Position type
    row = f"{'Type':<12}"
    for pv in found_players:
        pos_type = "Batter" if pv.position_type == "B" else "Pitcher"
        row += f" {pos_type:<15}"
    lines.append(row)

    # Total value
    row = f"{'Total Value':<12}"
    for pv in found_players:
        row += f" {pv.total_value:<15.2f}"
    lines.append(row)

    # Category values - collect all unique categories
    all_cats: dict[str, dict[str, tuple[float, float]]] = {}
    for pv in found_players:
        for cv in pv.category_values:
            cat_name = cv.category.value
            if cat_name not in all_cats:
                all_cats[cat_name] = {}
            all_cats[cat_name][pv.player_id] = (cv.raw_stat, cv.value)

    for cat_name, player_stats in all_cats.items():
        row = f"{cat_name:<12}"
        for pv in found_players:
            if pv.player_id in player_stats:
                raw, z = player_stats[pv.player_id]
                if cat_name in ("ERA", "WHIP"):
                    row += f" {raw:.2f} (z={z:.1f})  "
                else:
                    row += f" {raw:.0f} (z={z:.1f})   "
            else:
                row += f" {'-':<15}"
        lines.append(row)

    return "\n".join(lines)


def format_keeper_rankings(
    ranked: list,
    not_found: list[str],
    user_pick: int,
    teams: int,
    keeper_slots: int,
) -> str:
    """Format keeper rankings as a text table."""
    lines: list[str] = []

    if not_found:
        lines.append(f"Note: Could not find: {', '.join(not_found)}")
        lines.append("")

    lines.append(f"Keeper Rankings (Pick #{user_pick}, {teams} teams, {keeper_slots} keepers):")
    lines.append("")
    header = f"{'Rk':>4} {'Name':<25} {'Pos':<12} {'Value':>7} {'Repl':>7} {'Surplus':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, ks in enumerate(ranked, start=1):
        pos_str = "/".join(ks.eligible_positions) if ks.eligible_positions else "-"
        lines.append(
            f"{i:>4} {ks.name:<25} {pos_str:<12}"
            f" {ks.player_value:>7.1f} {ks.replacement_value:>7.1f}"
            f" {ks.surplus_value:>8.1f}"
        )

    return "\n".join(lines)
