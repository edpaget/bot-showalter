from datetime import date
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import (
    print_candidate_values,
    print_correlation_results,
    print_error,
)
from fantasy_baseball_manager.cli.factory import build_feature_context
from fantasy_baseball_manager.domain import FeatureCandidate
from fantasy_baseball_manager.services import aggregate_candidate

feature_app = typer.Typer(name="feature", help="Feature candidate exploration tools")


@feature_app.command("candidate")
def candidate_cmd(
    expression: Annotated[str, typer.Argument(help="SQL aggregation expression (e.g. 'AVG(launch_speed)')")],
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")] = ...,  # type: ignore[assignment]
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")] = ...,  # type: ignore[assignment]
    correlate: Annotated[bool, typer.Option("--correlate", help="Run target correlation scan")] = False,
    name: Annotated[str | None, typer.Option("--name", help="Save candidate with this name")] = None,
    min_pa: Annotated[int | None, typer.Option("--min-pa", help="Minimum plate appearances (batters)")] = None,
    min_ip: Annotated[float | None, typer.Option("--min-ip", help="Minimum innings pitched (pitchers)")] = None,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Aggregate statcast data with a SQL expression and optionally correlate with targets."""
    with build_feature_context(data_dir) as ctx:
        try:
            results = aggregate_candidate(
                ctx.statcast_conn,
                expression,
                season,
                player_type,
                min_pa=min_pa,
                min_ip=min_ip,
            )
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1) from e

        print_candidate_values(results)

        if correlate:
            try:
                scan_result = ctx.scanner.scan_target_correlations(expression, season, player_type)
            except ValueError as e:
                print_error(str(e))
                raise typer.Exit(code=1) from e
            print_correlation_results(scan_result)

        if name:
            candidate = FeatureCandidate(
                name=name,
                expression=expression,
                player_type=player_type,
                min_pa=min_pa,
                min_ip=min_ip,
                created_at=date.today().isoformat(),
            )
            ctx.candidate_repo.save(candidate)
            typer.echo(f"Saved candidate '{name}'")
