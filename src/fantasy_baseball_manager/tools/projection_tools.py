from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.tools._formatting import format_no_results


def create_lookup_projections_tool(container: AnalysisContainer) -> BaseTool:
    """Create a tool that looks up player projections."""

    @tool
    def lookup_projections(player_name: str, season: int, system: str | None = None) -> str:
        """Look up projections for a player. Optionally filter by projection system name."""
        projections = container.projection_lookup_service.lookup(player_name, season, system)
        if not projections:
            return format_no_results("projections", player_name=player_name, season=season, system=system)
        lines: list[str] = []
        for proj in projections:
            lines.append(f"{proj.player_name} — {proj.system} v{proj.version} ({proj.source_type}, {proj.player_type})")
            stat_parts: list[str] = []
            for stat_name in sorted(proj.stats):
                if stat_name.startswith("_"):
                    continue
                value = proj.stats[stat_name]
                if isinstance(value, float):
                    stat_parts.append(f"{stat_name}: {value:.3f}")
                else:
                    stat_parts.append(f"{stat_name}: {value}")
            lines.append(f"  {', '.join(stat_parts)}")
        return "\n".join(lines)

    return lookup_projections
