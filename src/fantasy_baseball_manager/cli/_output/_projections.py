from typing import TYPE_CHECKING

from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import PlayerProjection, SystemSummary

_METADATA_KEYS = {"_components", "_mode", "_pt_system", "rates"}


def print_player_projections(projections: list[PlayerProjection]) -> None:
    """Print player projection results."""
    if not projections:
        console.print("No projections found.")
        return
    for proj in projections:
        console.print(
            f"[bold]{proj.player_name}[/bold] — {proj.system} v{proj.version}"
            f" [dim]({proj.source_type}, {proj.player_type})[/dim]"
        )
        # Lineage: ensemble sources
        components = proj.stats.get("_components")
        if isinstance(components, dict):
            mode = proj.stats.get("_mode", "")
            parts = [f"{sys} {int(w * 100)}%" for sys, w in components.items()]
            console.print(f"  Sources: {', '.join(parts)} ({mode})")
        # Lineage: composite PT source
        pt_system = proj.stats.get("_pt_system")
        if isinstance(pt_system, str):
            console.print(f"  PT source: {pt_system}")
        # Stats table, filtering out metadata keys
        table = Table(show_header=False, show_edge=False, pad_edge=False, box=None)
        table.add_column("Stat")
        table.add_column("Value", justify="right")
        for stat_name in sorted(proj.stats):
            if stat_name in _METADATA_KEYS or stat_name.startswith("_"):
                continue
            value = proj.stats[stat_name]
            if isinstance(value, float):
                table.add_row(stat_name, f"{value:.3f}")
            else:
                table.add_row(stat_name, str(value))
        console.print(table)


def print_system_summaries(summaries: list[SystemSummary]) -> None:
    """Print a table of available projection systems."""
    if not summaries:
        console.print("No projection systems found for this season.")
        return
    table = Table(show_edge=False, pad_edge=False)
    table.add_column("System")
    table.add_column("Version")
    table.add_column("Source")
    table.add_column("Batters", justify="right")
    table.add_column("Pitchers", justify="right")
    table.add_column("Total", justify="right")
    for s in summaries:
        total = s.batter_count + s.pitcher_count
        table.add_row(s.system, s.version, s.source_type, str(s.batter_count), str(s.pitcher_count), str(total))
    console.print(table)
