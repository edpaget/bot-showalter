from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._defaults import _DataDirOpt, load_cli_defaults
from fantasy_baseball_manager.cli._output import (
    print_player_valuations,
    print_valuation_eval_result,
    print_valuation_rankings,
)
from fantasy_baseball_manager.cli.factory import (
    build_valuation_eval_context,
    build_valuations_context,
)
from fantasy_baseball_manager.config_league import load_league

valuations_app = typer.Typer(name="valuations", help="Look up and explore player valuations")


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
    version: Annotated[str | None, typer.Option("--version", help="Filter by valuation version")] = None,
    player_type: Annotated[str | None, typer.Option("--player-type", help="Filter by player type")] = None,
    position: Annotated[str | None, typer.Option("--position", help="Filter by position")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show valuation rankings as a leaderboard."""
    with build_valuations_context(data_dir) as ctx:
        results = ctx.lookup_service.rankings(
            season, system=system, player_type=player_type, position=position, top=top, version=version
        )
    print_valuation_rankings(results)


@valuations_app.command("evaluate")
def valuations_evaluate(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N mispricings")] = None,
    min_value: Annotated[
        float | None, typer.Option("--min-value", help="Min predicted or actual value to include")
    ] = None,
    top_n: Annotated[int | None, typer.Option("--top-n", help="Top N by predicted rank for population filter")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate valuation accuracy against end-of-season actuals."""
    defaults = load_cli_defaults()
    if system is None:
        system = defaults.system
    if version is None:
        version = defaults.version
    league = load_league(league_name, Path.cwd())
    with build_valuation_eval_context(data_dir) as ctx:
        result = ctx.evaluator.evaluate(system, version, season, league, top=top_n, min_value=min_value)
    print_valuation_eval_result(result, top=top)
