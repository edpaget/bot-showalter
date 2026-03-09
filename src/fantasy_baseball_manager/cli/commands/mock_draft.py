"""CLI commands for mock draft simulation."""

import dataclasses
import random
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._defaults import _DataDirOpt, load_cli_defaults
from fantasy_baseball_manager.cli._output import (
    print_availability_windows,
    print_batch_simulation_result,
    print_mock_draft_result,
    print_player_availability_curve,
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
    build_draft_roster_slots,
    compute_availability_windows,
    compute_player_availability_curve,
    run_batch_simulation,
    run_mock_draft,
)
from fantasy_baseball_manager.services.draft_plan import _user_pick_numbers

mock_app = typer.Typer(name="mock", help="Mock draft simulation")

_STRATEGY_MAP: dict[str, type] = {
    "adp": ADPBot,
    "best-value": BestValueBot,
    "positional-need": PositionalNeedBot,
    "random": RandomBot,
}


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
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    teams: Annotated[int, typer.Option("--teams", help="Number of teams (0=use league)")] = 0,
    position: Annotated[int, typer.Option("--position", help="Draft position (1-based, 0=random)")] = 0,
    strategy: Annotated[str, typer.Option("--strategy", help="User strategy")] = "best-value",
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Run a single mock draft and print the draft log."""
    defaults = load_cli_defaults()
    if system is None:
        system = defaults.system
    if version is None:
        version = defaults.version
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
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
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
    defaults = load_cli_defaults()
    if system is None:
        system = defaults.system
    if version is None:
        version = defaults.version
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
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
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
    defaults = load_cli_defaults()
    if system is None:
        system = defaults.system
    if version is None:
        version = defaults.version
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


@mock_app.command("availability")
def mock_availability(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    position: Annotated[int, typer.Option("--position", help="Draft position (1-based)")] = 1,
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    teams: Annotated[int, typer.Option("--teams", help="Number of teams (0=use league)")] = 0,
    simulations: Annotated[int, typer.Option("--simulations", help="Number of simulations")] = 100,
    player: Annotated[str | None, typer.Option("--player", help="Player name filter (substring)")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show player availability windows from mock draft simulations."""
    defaults = load_cli_defaults()
    if system is None:
        system = defaults.system
    if version is None:
        version = defaults.version
    board, league = _fetch_draft_board_data(season, system, version, league_name, provider, data_dir, None, None, None)

    if teams > 0:
        league = dataclasses.replace(league, teams=teams)

    draft_position = position - 1

    def user_factory(rng: random.Random) -> DraftBot:
        return _make_bot("best-value", rng)

    opponent_factories = [_make_opponent for _ in range(league.teams - 1)]

    result = run_batch_simulation(
        simulations,
        board,
        league,
        user_factory,
        opponent_factories,
        draft_position=draft_position,
        seed=seed,
    )

    player_names = {r.player_id: r.player_name for r in board.rows}
    player_positions = {r.player_id: r.position for r in board.rows}

    if player:
        # Find matching player via case-insensitive substring
        query = player.lower()
        matches = [
            (pid, name)
            for pid, name in player_names.items()
            if query in name.lower() and pid in result.all_player_picks
        ]
        if not matches:
            typer.echo(f"No player matching '{player}' found in simulation data.")
            raise typer.Exit(1)
        if len(matches) > 1:
            typer.echo(f"Multiple matches for '{player}':")
            for _, name in matches[:10]:
                typer.echo(f"  {name}")
            typer.echo("Please be more specific.")
            raise typer.Exit(1)

        pid, _ = matches[0]
        slots = build_draft_roster_slots(league)
        total_rounds = sum(slots.values())
        curve = compute_player_availability_curve(
            player_id=pid,
            all_player_picks=result.all_player_picks,
            player_names=player_names,
            player_positions=player_positions,
            n_simulations=simulations,
            slot=draft_position,
            teams=league.teams,
            total_rounds=total_rounds,
        )
        print_player_availability_curve(curve)
    else:
        # Compute user's first pick number for availability calculation
        slots = build_draft_roster_slots(league)
        total_rounds = sum(slots.values())
        user_picks = _user_pick_numbers(draft_position, league.teams, total_rounds)
        user_first_pick = user_picks[0]

        windows = compute_availability_windows(
            result.all_player_picks,
            player_names,
            player_positions,
            n_simulations=simulations,
            user_next_pick=user_first_pick,
        )
        print_availability_windows(windows)
