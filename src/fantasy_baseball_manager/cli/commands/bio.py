from typing import Annotated

import typer

from fantasy_baseball_manager.cli._defaults import _DataDirOpt  # noqa: TC001 — used at runtime by typer
from fantasy_baseball_manager.cli._output import print_player_summaries
from fantasy_baseball_manager.cli.factory import build_bio_context


def bio(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    team: Annotated[str | None, typer.Option("--team", help="Team name, abbreviation, or nickname")] = None,
    position: Annotated[str | None, typer.Option("--position", help="Filter by primary position")] = None,
    min_age: Annotated[int | None, typer.Option("--min-age", help="Minimum player age")] = None,
    max_age: Annotated[int | None, typer.Option("--max-age", help="Maximum player age")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Look up player biographies with optional team, position, and age filters."""
    with build_bio_context(data_dir) as ctx:
        resolved_team: str | None = None

        if team is not None:
            abbreviations = ctx.team_resolver.resolve(team)

            if not abbreviations:
                typer.echo(f"No team found matching '{team}'.", err=True)
                raise typer.Exit(code=1)

            if len(abbreviations) > 1:
                lines = [f"'{team}' matches multiple teams:"]
                for abbrev in abbreviations:
                    team_obj = ctx.team_repo.get_by_abbreviation(abbrev)
                    full_name = team_obj.name if team_obj else abbrev
                    lines.append(f"  {abbrev} ({full_name})")
                lines.append("Use a more specific name or abbreviation.")
                typer.echo("\n".join(lines), err=True)
                raise typer.Exit(code=1)

            resolved_team = abbreviations[0]
            if resolved_team.upper() != team.upper():
                typer.echo(f"\u2192 Resolved '{team}' to {resolved_team}")

        results = ctx.bio_service.find(
            season=season,
            team=resolved_team,
            position=position,
            min_age=min_age,
            max_age=max_age,
        )

    print_player_summaries(results)
