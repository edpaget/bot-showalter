"""CLI commands for mock draft simulation."""

import dataclasses
import random
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import (
    print_batch_simulation_result,
    print_mock_draft_result,
    print_strategy_comparison_table,
)
from fantasy_baseball_manager.cli.commands.draft import _fetch_draft_board_data
from fantasy_baseball_manager.domain import StrategyComparison
from fantasy_baseball_manager.services import (
    ADPBot,
    BestValueBot,
    DraftBot,
    PositionalNeedBot,
    RandomBot,
    run_batch_simulation,
    run_mock_draft,
)

mock_app = typer.Typer(name="mock", help="Mock draft simulation")

_STRATEGY_MAP: dict[str, type] = {
    "adp": ADPBot,
    "best-value": BestValueBot,
    "positional-need": PositionalNeedBot,
    "random": RandomBot,
}

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


def _make_bot(name: str, rng: random.Random) -> DraftBot:
    """Create a bot from a strategy name."""
    cls = _STRATEGY_MAP.get(name)
    if cls is None:
        msg = f"Unknown strategy: {name!r}. Valid: {', '.join(sorted(_STRATEGY_MAP))}"
        raise typer.BadParameter(msg)
    if cls in (ADPBot, BestValueBot):
        return cls(rng=rng, noise=0.0)
    return cls(rng=rng)


def _make_opponent(rng: random.Random) -> DraftBot:
    return ADPBot(rng=rng, noise=0.15)


@mock_app.command("single")
def mock_single(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    teams: Annotated[int, typer.Option("--teams", help="Number of teams (0=use league)")] = 0,
    position: Annotated[int, typer.Option("--position", help="Draft position (1-based, 0=random)")] = 0,
    strategy: Annotated[str, typer.Option("--strategy", help="User strategy")] = "best-value",
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Run a single mock draft and print the draft log."""
    board, league = _fetch_draft_board_data(season, system, version, league_name, provider, data_dir, None, None, None)

    if teams > 0:
        league = dataclasses.replace(league, teams=teams)

    num_teams = league.teams
    rng = random.Random(seed)  # noqa: S311

    user_bot = _make_bot(strategy, random.Random(rng.randint(0, 2**32)))  # noqa: S311
    user_idx = position - 1 if position > 0 else rng.randrange(num_teams)

    strategies: list[DraftBot] = []
    for i in range(num_teams):
        if i == user_idx:
            strategies.append(user_bot)
        else:
            strategies.append(_make_opponent(random.Random(rng.randint(0, 2**32))))  # noqa: S311

    result = run_mock_draft(board, league, strategies, seed=seed)
    print_mock_draft_result(result, user_team=user_idx)


@mock_app.command("batch")
def mock_batch(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    teams: Annotated[int, typer.Option("--teams", help="Number of teams (0=use league)")] = 0,
    position: Annotated[int, typer.Option("--position", help="Draft position (1-based, 0=random)")] = 0,
    strategy: Annotated[str, typer.Option("--strategy", help="User strategy")] = "best-value",
    simulations: Annotated[int, typer.Option("--simulations", help="Number of simulations")] = 100,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Run batch mock draft simulations and print aggregate statistics."""
    board, league = _fetch_draft_board_data(season, system, version, league_name, provider, data_dir, None, None, None)

    if teams > 0:
        league = dataclasses.replace(league, teams=teams)

    if strategy not in _STRATEGY_MAP:
        msg = f"Unknown strategy: {strategy!r}. Valid: {', '.join(sorted(_STRATEGY_MAP))}"
        raise typer.BadParameter(msg)

    def user_factory(rng: random.Random) -> DraftBot:
        return _make_bot(strategy, rng)

    opponent_factories = [_make_opponent for _ in range(league.teams - 1)]

    draft_position = position - 1 if position > 0 else None

    result = run_batch_simulation(
        simulations,
        board,
        league,
        user_factory,
        opponent_factories,
        draft_position=draft_position,
        seed=seed,
    )
    print_batch_simulation_result(result)


@mock_app.command("compare")
def mock_compare(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    teams: Annotated[int, typer.Option("--teams", help="Number of teams (0=use league)")] = 0,
    position: Annotated[int, typer.Option("--position", help="Draft position (1-based, 0=random)")] = 0,
    strategies: Annotated[
        str, typer.Option("--strategies", help="Comma-separated strategies")
    ] = "adp,best-value,positional-need",
    simulations: Annotated[int, typer.Option("--simulations", help="Number of simulations")] = 100,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Compare multiple strategies by running batch simulations for each."""
    board, league = _fetch_draft_board_data(season, system, version, league_name, provider, data_dir, None, None, None)

    if teams > 0:
        league = dataclasses.replace(league, teams=teams)

    strategy_names = [s.strip() for s in strategies.split(",")]
    for name in strategy_names:
        if name not in _STRATEGY_MAP:
            msg = f"Unknown strategy: {name!r}. Valid: {', '.join(sorted(_STRATEGY_MAP))}"
            raise typer.BadParameter(msg)

    draft_position = position - 1 if position > 0 else None
    opponent_factories = [_make_opponent for _ in range(league.teams - 1)]

    comparisons: list[StrategyComparison] = []
    for name in strategy_names:

        def user_factory(rng: random.Random, _name: str = name) -> DraftBot:
            return _make_bot(_name, rng)

        result = run_batch_simulation(
            simulations,
            board,
            league,
            user_factory,
            opponent_factories,
            draft_position=draft_position,
            seed=seed,
        )

        # Extract the user's comparison entry and rename to strategy name
        for comp in result.strategy_comparisons:
            if comp.strategy_name == "user":
                comparisons.append(
                    StrategyComparison(
                        strategy_name=name,
                        avg_value=comp.avg_value,
                        win_rate=comp.win_rate,
                    )
                )
                break

    print_strategy_comparison_table(comparisons)
