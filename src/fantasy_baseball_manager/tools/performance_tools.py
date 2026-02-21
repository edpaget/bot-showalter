from __future__ import annotations

from collections import defaultdict
from statistics import mean

from langchain_core.tools import BaseTool, tool

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.domain.performance_delta import PlayerStatDelta
from fantasy_baseball_manager.tools._formatting import format_no_results, format_table


def _aggregate_deltas(deltas: list[PlayerStatDelta]) -> dict[str, list[PlayerStatDelta]]:
    """Group deltas by player name."""
    by_player: dict[str, list[PlayerStatDelta]] = defaultdict(list)
    for d in deltas:
        by_player[d.player_name].append(d)
    return dict(by_player)


def _format_performance_table(
    player_scores: list[tuple[str, float, list[PlayerStatDelta]]],
) -> str:
    """Format aggregated performance data as a table."""
    headers = ["Player", "Score", "Stats"]
    alignments = ["l", "r", "l"]
    rows: list[list[str]] = []
    for name, score, stats in player_scores:
        stat_parts = [f"{d.stat_name}={d.performance_delta:+.3f}" for d in stats]
        rows.append([name, f"{score:+.3f}", ", ".join(stat_parts)])
    return format_table(headers, rows, alignments)


def create_get_overperformers_tool(container: AnalysisContainer) -> BaseTool:
    """Create a tool that finds players outperforming projections."""

    @tool
    def get_overperformers(
        system: str,
        version: str,
        season: int,
        player_type: str,
        top: int = 20,
    ) -> str:
        """Find players outperforming their projections, ranked by mean delta."""
        deltas = container.performance_report_service.compute_deltas(system, version, season, player_type)
        if not deltas:
            return format_no_results(
                "performance data", system=system, version=version, season=season, player_type=player_type
            )
        by_player = _aggregate_deltas(deltas)
        scored: list[tuple[str, float, list[PlayerStatDelta]]] = []
        for name, player_deltas in by_player.items():
            avg_delta = mean(d.performance_delta for d in player_deltas)
            scored.append((name, avg_delta, player_deltas))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top]
        return f"Overperformers ({system} v{version}, {player_type}s)\n{_format_performance_table(scored)}"

    return get_overperformers


def create_get_underperformers_tool(container: AnalysisContainer) -> BaseTool:
    """Create a tool that finds players underperforming projections."""

    @tool
    def get_underperformers(
        system: str,
        version: str,
        season: int,
        player_type: str,
        top: int = 20,
    ) -> str:
        """Find players underperforming their projections, ranked by mean delta (lowest first)."""
        deltas = container.performance_report_service.compute_deltas(system, version, season, player_type)
        if not deltas:
            return format_no_results(
                "performance data", system=system, version=version, season=season, player_type=player_type
            )
        by_player = _aggregate_deltas(deltas)
        scored: list[tuple[str, float, list[PlayerStatDelta]]] = []
        for name, player_deltas in by_player.items():
            avg_delta = mean(d.performance_delta for d in player_deltas)
            scored.append((name, avg_delta, player_deltas))
        scored.sort(key=lambda x: x[1])
        scored = scored[:top]
        return f"Underperformers ({system} v{version}, {player_type}s)\n{_format_performance_table(scored)}"

    return get_underperformers
