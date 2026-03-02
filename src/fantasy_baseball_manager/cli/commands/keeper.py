import csv
from pathlib import Path  # noqa: TC003 — Typer evaluates annotations at runtime
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import (
    print_adjusted_rankings,
    print_keeper_decisions,
    print_keeper_scenarios,
    print_keeper_solution,
    print_keeper_trade_impact,
    print_trade_evaluation,
)
from fantasy_baseball_manager.cli.factory import build_keeper_context
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain import Err, KeeperConstraints, KeeperDecision, Ok
from fantasy_baseball_manager.ingest import import_keeper_costs
from fantasy_baseball_manager.services import (
    compare_scenarios,
    compute_adjusted_valuations,
    compute_surplus,
    evaluate_trade,
    keeper_trade_impact,
    parse_league_keepers,
    set_keeper_cost,
    solve_keepers,
    solve_keepers_with_pool,
)

keeper_app = typer.Typer(name="keeper", help="Keeper league cost management")


@keeper_app.command("import")
def import_cmd(
    csv_path: Annotated[Path, typer.Argument(help="Path to keeper costs CSV file")],
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    source: Annotated[str, typer.Option(help="Cost source type")] = "auction",
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Import keeper costs from a CSV file."""
    if not csv_path.exists():
        typer.echo(f"Error: file not found: {csv_path}", err=True)
        raise typer.Exit(code=1)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with build_keeper_context(data_dir) as ctx:
        players = ctx.player_repo.all()
        result = import_keeper_costs(
            rows, ctx.keeper_repo, players, season=season, league=league, default_source=source
        )
        ctx.conn.commit()

    typer.echo(f"Loaded {result.loaded} keeper costs, skipped {result.skipped}")
    if result.unmatched:
        typer.echo(f"Unmatched players ({len(result.unmatched)}): {', '.join(result.unmatched)}")


@keeper_app.command("set")
def set_cmd(
    player_name: Annotated[str, typer.Argument(help="Player name to search for")],
    cost: Annotated[float, typer.Option(help="Keeper cost")],
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    years: Annotated[int, typer.Option(help="Years remaining on contract")] = 1,
    source: Annotated[str, typer.Option(help="Cost source type")] = "auction",
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Set a keeper cost for a single player."""
    with build_keeper_context(data_dir) as ctx:
        result = set_keeper_cost(player_name, cost, season, league, ctx.player_repo, ctx.keeper_repo, years, source)
        match result:
            case Ok(kc):
                ctx.conn.commit()
                player = ctx.player_repo.get_by_id(kc.player_id)
                name = f"{player.name_first} {player.name_last}" if player else "Unknown"
                typer.echo(f"Set keeper cost for {name}: ${cost:.0f} ({source}, {years}yr)")
            case Err(msg):
                typer.echo(f"Error: {msg}", err=True)
                raise typer.Exit(code=1)


@keeper_app.command("decisions")
def decisions_cmd(
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option(help="Valuation system name")],
    threshold: Annotated[float, typer.Option(help="Minimum surplus for keep recommendation")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor for multi-year surplus")] = 0.85,
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Show keeper decisions ranked by surplus value."""
    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return
        valuations = ctx.valuation_repo.get_by_season(season, system)
        players = ctx.player_repo.all()
        decisions = compute_surplus(keeper_costs, valuations, players, threshold=threshold, decay=decay)

    print_keeper_decisions(decisions)


@keeper_app.command("adjusted-rankings")
def adjusted_rankings_cmd(
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name (must match fbm.toml)")],
    system: Annotated[str, typer.Option(help="Valuation system name")],
    threshold: Annotated[float, typer.Option(help="Minimum surplus for keep recommendation")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor for multi-year surplus")] = 0.85,
    top: Annotated[int | None, typer.Option(help="Show only top N players")] = None,
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Show post-keeper adjusted rankings with recalculated replacement levels."""
    league_settings = load_league(league, Path.cwd())

    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return

        valuations = ctx.valuation_repo.get_by_season(season, system)
        players = ctx.player_repo.all()
        decisions = compute_surplus(keeper_costs, valuations, players, threshold=threshold, decay=decay)
        kept_player_ids = {d.player_id for d in decisions if d.recommendation == "keep"}

        if not kept_player_ids:
            typer.echo("No players recommended for keeping. Adjusted rankings match original.")
            return

        # Determine projection system from valuations
        if not valuations:
            typer.echo("No valuations found for the specified season and system.")
            return
        proj_system = valuations[0].projection_system

        projections = ctx.projection_repo.get_by_season(season, system=proj_system)
        batter_positions = ctx.eligibility_service.get_batter_positions(season, league_settings)
        pitcher_ids = [p.player_id for p in projections if p.player_type == "pitcher"]
        pitcher_positions = ctx.eligibility_service.get_pitcher_positions(season, league_settings, pitcher_ids)

        results = compute_adjusted_valuations(
            kept_player_ids=kept_player_ids,
            projections=projections,
            batter_positions=batter_positions,
            pitcher_positions=pitcher_positions,
            league=league_settings,
            original_valuations=valuations,
            players=players,
        )

    print_adjusted_rankings(results, top=top)


@keeper_app.command("trade-eval")
def trade_eval_cmd(
    gives: Annotated[list[str], typer.Option("--gives", help="Player(s) you give away")],
    receives: Annotated[list[str], typer.Option("--receives", help="Player(s) you receive")],
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option(help="Valuation system name")],
    decay: Annotated[float, typer.Option(help="Decay factor")] = 0.85,
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Evaluate a trade using keeper surplus value."""
    with build_keeper_context(data_dir) as ctx:
        give_ids: list[int] = []
        for name in gives:
            matches = ctx.player_repo.search_by_name(name)
            if len(matches) == 0:
                typer.echo(f"Error: no player found matching '{name}'", err=True)
                raise typer.Exit(code=1)
            if len(matches) > 1:
                names = [f"{p.name_first} {p.name_last}" for p in matches]
                typer.echo(f"Error: ambiguous name '{name}', matches: {', '.join(names)}", err=True)
                raise typer.Exit(code=1)
            assert matches[0].id is not None  # noqa: S101
            give_ids.append(matches[0].id)

        receive_ids: list[int] = []
        for name in receives:
            matches = ctx.player_repo.search_by_name(name)
            if len(matches) == 0:
                typer.echo(f"Error: no player found matching '{name}'", err=True)
                raise typer.Exit(code=1)
            if len(matches) > 1:
                names = [f"{p.name_first} {p.name_last}" for p in matches]
                typer.echo(f"Error: ambiguous name '{name}', matches: {', '.join(names)}", err=True)
                raise typer.Exit(code=1)
            assert matches[0].id is not None  # noqa: S101
            receive_ids.append(matches[0].id)

        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        valuations = ctx.valuation_repo.get_by_season(season, system)
        players = ctx.player_repo.all()

    result = evaluate_trade(give_ids, receive_ids, keeper_costs, valuations, players, decay)
    print_trade_evaluation(result)


def _resolve_player_id(name: str, player_repo: object) -> int:
    """Resolve a player name to a single ID, or exit with an error."""
    matches = player_repo.search_by_name(name)  # type: ignore[union-attr]
    if len(matches) == 0:
        typer.echo(f"Error: no player found matching '{name}'", err=True)
        raise typer.Exit(code=1)
    if len(matches) > 1:
        names = [f"{p.name_first} {p.name_last}" for p in matches]
        typer.echo(f"Error: ambiguous name '{name}', matches: {', '.join(names)}", err=True)
        raise typer.Exit(code=1)
    assert matches[0].id is not None  # noqa: S101
    return matches[0].id


def _parse_position_limits(raw: list[str]) -> dict[str, int]:
    """Parse 'pos=N' strings into a dict."""
    limits: dict[str, int] = {}
    for item in raw:
        pos, _, count = item.partition("=")
        limits[pos.strip().lower()] = int(count.strip())
    return limits


@keeper_app.command("optimize")
def optimize_cmd(
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option(help="Valuation system name")],
    max_keepers: Annotated[int, typer.Option(help="Maximum number of keepers")],
    max_per_position: Annotated[list[str] | None, typer.Option(help="Position limits, e.g. 'c=1'")] = None,
    max_cost: Annotated[float | None, typer.Option(help="Maximum total keeper cost")] = None,
    required: Annotated[list[str] | None, typer.Option(help="Player names that must be kept")] = None,
    league_keepers: Annotated[
        Path | None, typer.Option(help="CSV of other teams' keepers for pool-adjusted mode")
    ] = None,
    threshold: Annotated[float, typer.Option(help="Minimum surplus for candidates")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor")] = 0.85,
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Find the optimal keeper set maximizing total surplus."""
    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return
        valuations = ctx.valuation_repo.get_by_season(season, system)
        players = ctx.player_repo.all()
        candidates = compute_surplus(keeper_costs, valuations, players, threshold=threshold, decay=decay)

        required_ids: list[int] = []
        if required:
            for name in required:
                required_ids.append(_resolve_player_id(name, ctx.player_repo))

        pos_limits = _parse_position_limits(max_per_position) if max_per_position else None
        constraints = KeeperConstraints(
            max_keepers=max_keepers,
            max_per_position=pos_limits,
            max_cost=max_cost,
            required_keepers=required_ids,
        )

        if league_keepers is not None:
            with open(league_keepers, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            league_ids, _unmatched = parse_league_keepers(rows, players)
            league_settings = load_league(league, Path.cwd())
            solution = solve_keepers_with_pool(candidates, constraints, league_ids, valuations, league_settings)
        else:
            solution = solve_keepers(candidates, constraints)

    print_keeper_solution(solution)


@keeper_app.command("scenario")
def scenario_cmd(
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option(help="Valuation system name")],
    max_keepers: Annotated[int, typer.Option(help="Maximum number of keepers")],
    scenario: Annotated[list[str], typer.Option(help="Scenario as 'Name:Player1,Player2'")],
    threshold: Annotated[float, typer.Option(help="Minimum surplus for candidates")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor")] = 0.85,
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Compare named keeper scenarios side-by-side."""
    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return
        valuations = ctx.valuation_repo.get_by_season(season, system)
        players = ctx.player_repo.all()
        candidates = compute_surplus(keeper_costs, valuations, players, threshold=threshold, decay=decay)

        parsed: list[tuple[str, list[int]]] = []
        for s in scenario:
            name, _, player_names_str = s.partition(":")
            player_names = [n.strip() for n in player_names_str.split(",")]
            ids: list[int] = []
            for pname in player_names:
                ids.append(_resolve_player_id(pname, ctx.player_repo))
            parsed.append((name.strip(), ids))

        constraints = KeeperConstraints(max_keepers=max_keepers)
        result = compare_scenarios(parsed, candidates, constraints)

    print_keeper_scenarios(result)


@keeper_app.command("trade-impact")
def trade_impact_cmd(
    season: Annotated[int, typer.Option(help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option(help="Valuation system name")],
    max_keepers: Annotated[int, typer.Option(help="Maximum number of keepers")],
    acquire: Annotated[list[str] | None, typer.Option(help="Player names to acquire")] = None,
    release: Annotated[list[str] | None, typer.Option(help="Player names to release")] = None,
    acquire_cost: Annotated[list[float] | None, typer.Option(help="Keeper cost for each acquired player")] = None,
    threshold: Annotated[float, typer.Option(help="Minimum surplus for candidates")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor")] = 0.85,
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
) -> None:
    """Show how acquiring/releasing players changes the optimal keeper set."""
    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return
        valuations = ctx.valuation_repo.get_by_season(season, system)
        players = ctx.player_repo.all()
        candidates = compute_surplus(keeper_costs, valuations, players, threshold=threshold, decay=decay)

        # Build valuation lookup for acquired players
        val_lookup: dict[int, float] = {}
        val_pos: dict[int, str] = {}
        for v in valuations:
            if v.player_id not in val_lookup or v.value > val_lookup[v.player_id]:
                val_lookup[v.player_id] = v.value
                val_pos[v.player_id] = v.position

        acquire_decisions: list[KeeperDecision] = []
        if acquire:
            costs = acquire_cost or [0.0] * len(acquire)
            for i, name in enumerate(acquire):
                pid = _resolve_player_id(name, ctx.player_repo)
                cost = costs[i] if i < len(costs) else 0.0
                value = val_lookup.get(pid, 0.0)
                pos = val_pos.get(pid, "util")
                player = ctx.player_repo.get_by_id(pid)
                pname = f"{player.name_first} {player.name_last}" if player else name
                acquire_decisions.append(
                    KeeperDecision(
                        player_id=pid,
                        player_name=pname,
                        position=pos,
                        cost=cost,
                        projected_value=value,
                        surplus=value - cost,
                        years_remaining=1,
                        recommendation="keep" if value - cost >= 0 else "release",
                    )
                )

        release_ids: list[int] = []
        if release:
            for name in release:
                release_ids.append(_resolve_player_id(name, ctx.player_repo))

        constraints = KeeperConstraints(max_keepers=max_keepers)
        impact = keeper_trade_impact(candidates, constraints, acquire_decisions, release_ids)

    print_keeper_trade_impact(impact)
