from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import (
    print_player_valuations,
    print_valuation_eval_result,
    print_valuation_rankings,
)
from fantasy_baseball_manager.cli.factory import build_valuation_eval_context, build_valuations_context
from fantasy_baseball_manager.config_league import load_league

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
def valuations_rankings(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str | None, typer.Option("--system", help="Filter by valuation system")] = None,
    player_type: Annotated[str | None, typer.Option("--player-type", help="Filter by player type")] = None,
    position: Annotated[str | None, typer.Option("--position", help="Filter by position")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show valuation rankings as a leaderboard."""
    with build_valuations_context(data_dir) as ctx:
        results = ctx.lookup_service.rankings(
            season, system=system, player_type=player_type, position=position, top=top
        )
    print_valuation_rankings(results)


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
