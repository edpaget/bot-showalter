from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.tools._formatting import format_table


def create_get_value_over_adp_tool(container: AnalysisContainer) -> BaseTool:
    """Create a tool that computes value-over-ADP analysis."""

    @tool
    def get_value_over_adp(
        season: int,
        system: str,
        version: str,
        player_type: str | None = None,
        top: int = 10,
    ) -> str:
        """Compare player valuations against ADP to find buy targets, overvalued players, and sleepers."""
        report = container.adp_report_service.compute_value_over_adp(
            season=season,
            system=system,
            version=version,
            player_type=player_type,
            top=top,
        )
        sections: list[str] = []
        if report.buy_targets:
            headers = ["Player", "ZAR Rank", "ZAR Value", "ADP Rank", "Delta"]
            alignments = ["l", "r", "r", "r", "r"]
            rows: list[list[str]] = []
            for entry in report.buy_targets:
                rows.append(
                    [
                        entry.player_name,
                        str(entry.zar_rank),
                        f"${entry.zar_value:.1f}",
                        str(entry.adp_rank),
                        f"+{entry.rank_delta}",
                    ]
                )
            sections.append(f"Buy Targets (undervalued vs ADP)\n{format_table(headers, rows, alignments)}")

        if report.avoid_list:
            headers = ["Player", "ZAR Rank", "ZAR Value", "ADP Rank", "Delta"]
            alignments = ["l", "r", "r", "r", "r"]
            rows = []
            for entry in report.avoid_list:
                rows.append(
                    [
                        entry.player_name,
                        str(entry.zar_rank),
                        f"${entry.zar_value:.1f}",
                        str(entry.adp_rank),
                        str(entry.rank_delta),
                    ]
                )
            sections.append(f"Avoid List (overvalued vs ADP)\n{format_table(headers, rows, alignments)}")

        if report.unranked_valuable:
            headers = ["Player", "ZAR Rank", "ZAR Value"]
            alignments = ["l", "r", "r"]
            rows = []
            for entry in report.unranked_valuable:
                rows.append(
                    [
                        entry.player_name,
                        str(entry.zar_rank),
                        f"${entry.zar_value:.1f}",
                    ]
                )
            sections.append(f"Unranked Sleepers (no ADP but highly valued)\n{format_table(headers, rows, alignments)}")

        if not sections:
            return "No value-over-ADP data found."
        return "\n\n".join(sections)

    return get_value_over_adp
