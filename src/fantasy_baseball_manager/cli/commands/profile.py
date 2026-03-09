from typing import Annotated

import typer

from fantasy_baseball_manager.cli._defaults import _DataDirOpt  # noqa: TC001 — used at runtime by typer
from fantasy_baseball_manager.cli._output import (
    print_column_profiles,
    print_column_ranking,
    print_correlation_results,
    print_error,
    print_stability_result,
)
from fantasy_baseball_manager.cli.factory import build_profile_context
from fantasy_baseball_manager.services import NUMERIC_COLUMNS, rank_columns

profile_app = typer.Typer(name="profile", help="Profile statcast data distributions")


@profile_app.command("columns")
def profile_columns_cmd(  # pragma: no cover
    columns: Annotated[list[str] | None, typer.Argument(help="Column name(s) to profile")] = None,
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")] = ...,  # type: ignore[assignment]
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")] = ...,  # type: ignore[assignment]
    all_columns: Annotated[bool, typer.Option("--all", help="Profile all numeric columns")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Profile distribution statistics for statcast columns at the player-season level."""
    if all_columns:
        target_columns = list(NUMERIC_COLUMNS)
    elif columns:
        target_columns = columns
    else:
        print_error("Provide column name(s) or use --all to profile all numeric columns.")
        raise typer.Exit(code=1)

    with build_profile_context(data_dir) as ctx:
        try:
            profiles = ctx.profiler.profile_columns(target_columns, season, player_type)
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1) from e

    print_column_profiles(profiles)


@profile_app.command("correlate")
def correlate_cmd(  # pragma: no cover
    columns: Annotated[list[str], typer.Argument(help="Column spec(s) to correlate")],
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")] = ...,  # type: ignore[assignment]
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")] = ...,  # type: ignore[assignment]
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Correlate statcast columns against model targets."""
    with build_profile_context(data_dir) as ctx:
        try:
            results = ctx.scanner.scan_multiple(columns, season, player_type)
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1) from e

    for result in results:
        print_correlation_results(result)

    if len(results) > 1:
        rankings = rank_columns(results)
        print_column_ranking(rankings)


@profile_app.command("stability")
def stability_cmd(
    column: Annotated[str, typer.Argument(help="Column spec to check")],
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")] = ...,  # type: ignore[assignment]
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")] = ...,  # type: ignore[assignment]
    target: Annotated[str | None, typer.Option("--target", help="Single target to check")] = None,
    all_targets: Annotated[bool, typer.Option("--all-targets", help="Check all targets")] = False,
    exclude_season: Annotated[list[int] | None, typer.Option("--exclude-season", help="Season(s) to exclude")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Check temporal stability of feature-target correlations across seasons."""
    if target is None and not all_targets:
        print_error("Provide --target <name> or --all-targets.")
        raise typer.Exit(code=1)

    effective_seasons = season
    if exclude_season:
        effective_seasons = [s for s in season if s not in exclude_season]

    check_target = target if not all_targets else None

    with build_profile_context(data_dir) as ctx:
        try:
            result = ctx.stability_checker.check_temporal_stability(
                column, check_target, effective_seasons, player_type
            )
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1) from e

    print_stability_result(result)
