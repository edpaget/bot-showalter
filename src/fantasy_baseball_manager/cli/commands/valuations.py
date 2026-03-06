from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import (
    print_error,
    print_player_valuations,
    print_valuation_eval_result,
    print_valuation_rankings,
)
from fantasy_baseball_manager.cli.factory import (
    build_injury_adjusted_valuations_context,
    build_valuation_eval_context,
    build_valuations_context,
    create_model,
)
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain import Err
from fantasy_baseball_manager.services import compute_injury_adjusted_valuations_list

valuations_app = typer.Typer(name="valuations", help="Look up and explore player valuations")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


@valuations_app.command("lookup")
def valuations_lookup(
    player_name: Annotated[str, typer.Argument(help="Player name ('Last' or 'Last, First')")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str | None, typer.Option("--system", help="Filter by valuation system")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Look up a player's valuations across systems."""
    with build_valuations_context(data_dir) as ctx:
        results = ctx.lookup_service.lookup(player_name, season, system=system)
    print_player_valuations(results)


@valuations_app.command("rankings")
def valuations_rankings(  # pragma: no cover
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str | None, typer.Option("--system", help="Filter by valuation system")] = None,
    player_type: Annotated[str | None, typer.Option("--player-type", help="Filter by player type")] = None,
    position: Annotated[str | None, typer.Option("--position", help="Filter by position")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    injury_adjusted: Annotated[bool, typer.Option("--injury-adjusted", help="Apply injury risk discount")] = False,
    league_name: Annotated[
        str | None, typer.Option("--league", help="League name (required w/ --injury-adjusted)")
    ] = None,
    projection_system: Annotated[str | None, typer.Option("--projection-system", help="Projection system")] = None,
    projection_version: Annotated[str | None, typer.Option("--projection-version", help="Projection version")] = None,
    model_name: Annotated[str, typer.Option("--model", help="Valuation model")] = "zar",
    seasons_back: Annotated[int, typer.Option("--seasons-back", help="Lookback window for injury data")] = 5,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show valuation rankings as a leaderboard."""
    if not injury_adjusted:
        with build_valuations_context(data_dir) as ctx:
            results = ctx.lookup_service.rankings(
                season, system=system, player_type=player_type, position=position, top=top
            )
        print_valuation_rankings(results)
        return

    if league_name is None or projection_system is None:
        print_error("--league and --projection-system are required with --injury-adjusted")
        raise typer.Exit(code=1)

    league = load_league(league_name, Path.cwd())
    season_list = list(range(season - seasons_back + 1, season + 1))

    with build_injury_adjusted_valuations_context(data_dir) as ctx:
        result = create_model(
            model_name,
            projection_repo=ctx.projection_repo,
            position_repo=ctx.projection_repo,
            player_repo=ctx.player_repo,
            eligibility_service=ctx.eligibility_service,
        )
        if isinstance(result, Err):
            print_error(result.error.message)
            raise typer.Exit(code=1)
        model = result.value

        valuations = compute_injury_adjusted_valuations_list(
            season=season,
            league=league,
            projection_system=projection_system,
            projection_version=projection_version,
            season_list=season_list,
            profiler=ctx.profiler,
            model=model,  # type: ignore[arg-type]  # Model satisfies Predictable structurally
            player_repo=ctx.player_repo,
        )

    if player_type is not None:
        valuations = [v for v in valuations if v.player_type == player_type]
    if position is not None:
        valuations = [v for v in valuations if v.position == position]
    if top is not None:
        valuations = valuations[:top]

    print_valuation_rankings(valuations)


@valuations_app.command("evaluate")
def valuations_evaluate(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = "1.0",
    top: Annotated[int | None, typer.Option("--top", help="Show top N mispricings")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate valuation accuracy against end-of-season actuals."""
    league = load_league(league_name, Path.cwd())
    with build_valuation_eval_context(data_dir) as ctx:
        result = ctx.evaluator.evaluate(system or "zar", version or "1.0", season, league)
    print_valuation_eval_result(result, top=top)
