from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from fantasy_baseball_manager.analysis_container import AnalysisContainer
from fantasy_baseball_manager.tools._formatting import format_no_results, format_player_summary_table


def create_search_players_tool(container: AnalysisContainer) -> BaseTool:
    """Create a tool that searches for players by name."""

    @tool
    def search_players(name: str, season: int) -> str:
        """Search for players by name. Returns a summary table of matching players."""
        results = container.player_bio_service.search(name, season)
        if not results:
            return format_no_results("players", name=name, season=season)
        return format_player_summary_table(results)

    return search_players


def create_get_player_bio_tool(container: AnalysisContainer) -> BaseTool:
    """Create a tool that looks up a player's biography."""

    @tool
    def get_player_bio(player_name: str, season: int) -> str:
        """Look up a player's bio. Returns details for one match, or a list if ambiguous."""
        results = container.player_bio_service.search(player_name, season)
        if not results:
            return format_no_results("players", name=player_name, season=season)
        if len(results) > 1:
            table = format_player_summary_table(results)
            return f"Multiple players found for '{player_name}':\n{table}"
        bio = results[0]
        lines = [
            bio.name,
            f"  Team: {bio.team}",
            f"  Age: {bio.age}" if bio.age is not None else "  Age: unknown",
            f"  Position: {bio.primary_position}",
            f"  Bats/Throws: {bio.bats or '?'}/{bio.throws or '?'}",
            f"  Experience: {bio.experience} years",
        ]
        return "\n".join(lines)

    return get_player_bio


def create_find_players_tool(container: AnalysisContainer) -> BaseTool:
    """Create a tool that finds players by various filters."""

    @tool
    def find_players(
        season: int,
        team: str | None = None,
        min_age: int | None = None,
        max_age: int | None = None,
        min_experience: int | None = None,
        position: str | None = None,
    ) -> str:
        """Find players matching filter criteria like team, age range, experience, and position."""
        results = container.player_bio_service.find(
            season=season,
            team=team,
            min_age=min_age,
            max_age=max_age,
            min_experience=min_experience,
            position=position,
        )
        if not results:
            return format_no_results(
                "players",
                season=season,
                team=team,
                min_age=min_age,
                max_age=max_age,
                min_experience=min_experience,
                position=position,
            )
        return format_player_summary_table(results)

    return find_players
