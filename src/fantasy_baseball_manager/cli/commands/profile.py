from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import print_column_profiles, print_error
from fantasy_baseball_manager.cli.factory import build_profile_context
from fantasy_baseball_manager.services import NUMERIC_COLUMNS

profile_app = typer.Typer(name="profile", help="Profile statcast data distributions")


@profile_app.command("columns")
def profile_columns_cmd(
    columns: Annotated[list[str] | None, typer.Argument(help="Column name(s) to profile")] = None,
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")] = ...,  # type: ignore[assignment]
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")] = ...,  # type: ignore[assignment]
    all_columns: Annotated[bool, typer.Option("--all", help="Profile all numeric columns")] = False,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
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
