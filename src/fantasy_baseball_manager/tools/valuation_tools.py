from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.tools._formatting import format_no_results, format_table


def create_lookup_valuations_tool(container: AnalysisContainer) -> BaseTool:
    """Create a tool that looks up player valuations."""

    @tool
    def lookup_valuations(player_name: str, season: int, system: str | None = None) -> str:
        """Look up valuations for a player. Optionally filter by valuation system name."""
        valuations = container.valuation_lookup_service.lookup(player_name, season, system)
        if not valuations:
            return format_no_results("valuations", player_name=player_name, season=season, system=system)
        lines: list[str] = []
        for val in valuations:
            lines.append(f"{val.player_name} — {val.system} v{val.version} ({val.player_type}, {val.position})")
            lines.append(f"  Projection: {val.projection_system} v{val.projection_version}")
            lines.append(f"  Value: ${val.value:.2f}  Rank: {val.rank}")
            if val.category_scores:
                cats = ", ".join(f"{k}={v:.2f}" for k, v in sorted(val.category_scores.items()))
                lines.append(f"  Categories: {cats}")
        return "\n".join(lines)

    return lookup_valuations


def create_get_rankings_tool(container: AnalysisContainer) -> BaseTool:
    """Create a tool that returns valuation rankings."""

    @tool
    def get_rankings(
        season: int,
        system: str,
        player_type: str | None = None,
        position: str | None = None,
        top: int = 20,
    ) -> str:
        """Get player rankings by valuation system. Optionally filter by player type and position."""
        rankings = container.valuation_lookup_service.rankings(
            season=season,
            system=system,
            player_type=player_type,
            position=position,
            top=top,
        )
        if not rankings:
            return format_no_results(
                "rankings", season=season, system=system, player_type=player_type, position=position
            )
        headers = ["Rank", "Player", "Type", "Pos", "Value"]
        alignments = ["r", "l", "l", "l", "r"]
        rows: list[list[str]] = []
        for val in rankings:
            rows.append(
                [
                    str(val.rank),
                    val.player_name,
                    val.player_type,
                    val.position,
                    f"${val.value:.2f}",
                ]
            )
        return format_table(headers, rows, alignments)

    return get_rankings
