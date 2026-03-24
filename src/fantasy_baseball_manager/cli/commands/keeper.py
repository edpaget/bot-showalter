import csv
from collections.abc import Callable  # noqa: TC003 — used in non-Typer annotations
from pathlib import Path  # noqa: TC003 — Typer evaluates annotations at runtime
from typing import TYPE_CHECKING, Annotated

import typer

from fantasy_baseball_manager.cli._defaults import _DataDirOpt, load_cli_defaults
from fantasy_baseball_manager.cli._output import (
    print_adjusted_rankings,
    print_keeper_decisions,
    print_keeper_scenarios,
    print_keeper_solution,
    print_keeper_trade_impact,
    print_league_keepers,
    print_trade_evaluation,
)
from fantasy_baseball_manager.cli.factory import build_keeper_context
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain import Err, KeeperConstraints, KeeperDecision, LeagueKeeper, Ok, PlayerType
from fantasy_baseball_manager.ingest import import_keeper_costs, import_league_keepers
from fantasy_baseball_manager.name_utils import resolve_players
from fantasy_baseball_manager.services import (
    compare_scenarios,
    compute_adjusted_valuations,
    compute_pick_value_curve,
    compute_surplus,
    evaluate_trade,
    keeper_trade_impact,
    parse_league_keepers,
    round_to_dollar_cost,
    set_keeper_cost,
    solve_keepers,
    solve_keepers_with_pool,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.cli.factory import KeeperContext
    from fantasy_baseball_manager.repos import PlayerRepo

keeper_app = typer.Typer(name="keeper", help="Keeper league cost management")


def _build_cost_translator(
    ctx: KeeperContext,
    season: int,
    league_name: str,
    system: str,
    provider: str,
    version: str = "production",
) -> Callable[[int], float]:
    """Build a round→dollar cost translator using the pick value curve."""
    adp = ctx.adp_repo.get_by_season(season, provider=provider)
    valuations = ctx.valuation_repo.get_by_season(season, system, version=version)
    league_settings = load_league(league_name, Path.cwd())
    curve = compute_pick_value_curve(adp, valuations, league_settings)
    return lambda round_num: round_to_dollar_cost(round_num, league_settings, curve)


@keeper_app.command("import")
def import_cmd(
    csv_path: Annotated[Path, typer.Argument(help="Path to keeper costs CSV file")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    source: Annotated[str, typer.Option(help="Cost source type")] = "auction",
    fmt: Annotated[str, typer.Option("--format", help="Cost format: auction or draft-pick")] = "auction",
    system: Annotated[str | None, typer.Option("--system", help="Valuation system (required for draft-pick)")] = None,
    provider: Annotated[str | None, typer.Option(help="ADP provider (required for draft-pick)")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Import keeper costs from a CSV file."""
    defaults = load_cli_defaults()
    if version is None:
        version = defaults.version
    if not csv_path.exists():
        typer.echo(f"Error: file not found: {csv_path}", err=True)
        raise typer.Exit(code=1)

    if fmt == "draft-pick":
        if system is None:
            typer.echo("Error: --system is required for draft-pick format", err=True)
            raise typer.Exit(code=1)
        if provider is None:
            typer.echo("Error: --provider is required for draft-pick format", err=True)
            raise typer.Exit(code=1)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with build_keeper_context(data_dir) as ctx:
        cost_translator: Callable[[int], float] | None = None
        if fmt == "draft-pick":
            assert system is not None  # noqa: S101
            assert provider is not None  # noqa: S101
            cost_translator = _build_cost_translator(ctx, season, league, system, provider, version)

        players = ctx.player_repo.all()
        result = import_keeper_costs(
            rows,
            ctx.keeper_repo,
            players,
            season=season,
            league=league,
            default_source=source,
            cost_translator=cost_translator,
        )
        ctx.conn.commit()

    typer.echo(f"Loaded {result.loaded} keeper costs, skipped {result.skipped}")
    if result.unmatched:
        typer.echo(f"Unmatched players ({len(result.unmatched)}): {', '.join(result.unmatched)}")


@keeper_app.command("set")
def set_cmd(
    player_name: Annotated[str, typer.Argument(help="Player name to search for")],
    cost: Annotated[float | None, typer.Option(help="Keeper cost")] = None,
    round_num: Annotated[int | None, typer.Option("--round", help="Draft round (converted to dollars)")] = None,
    season: Annotated[int | None, typer.Option("--season", help="Season year")] = None,
    league: Annotated[str, typer.Option(help="League name")] = "dynasty",
    years: Annotated[int, typer.Option(help="Years remaining on contract")] = 1,
    source: Annotated[str, typer.Option(help="Cost source type")] = "auction",
    system: Annotated[str | None, typer.Option("--system", help="Valuation system (required for --round)")] = None,
    provider: Annotated[str | None, typer.Option(help="ADP provider (required for --round)")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Set a keeper cost for a single player."""
    defaults = load_cli_defaults()
    if season is None:
        season = defaults.season
    if version is None:
        version = defaults.version
    if cost is not None and round_num is not None:
        typer.echo("Error: --cost and --round are mutually exclusive", err=True)
        raise typer.Exit(code=1)
    if cost is None and round_num is None:
        typer.echo("Error: one of --cost or --round is required", err=True)
        raise typer.Exit(code=1)

    if round_num is not None:
        if system is None:
            typer.echo("Error: --system is required when using --round", err=True)
            raise typer.Exit(code=1)
        if provider is None:
            typer.echo("Error: --provider is required when using --round", err=True)
            raise typer.Exit(code=1)

    with build_keeper_context(data_dir) as ctx:
        if round_num is not None:
            assert system is not None  # noqa: S101
            assert provider is not None  # noqa: S101
            translator = _build_cost_translator(ctx, season, league, system, provider, version)
            effective_cost = translator(round_num)
            effective_source = "draft_round"
            original_round: int | None = round_num
        else:
            assert cost is not None  # noqa: S101
            effective_cost = cost
            effective_source = source
            original_round = None

        result = set_keeper_cost(
            player_name,
            effective_cost,
            season,
            league,
            ctx.player_repo,
            ctx.keeper_repo,
            years,
            effective_source,
            original_round=original_round,
        )
        match result:
            case Ok(kc):
                ctx.conn.commit()
                player = ctx.player_repo.get_by_id(kc.player_id)
                name = f"{player.name_first} {player.name_last}" if player else "Unknown"
                if kc.original_round is not None:
                    typer.echo(f"Set keeper cost for {name}: Round {kc.original_round} (~${kc.cost:.0f})")
                else:
                    typer.echo(f"Set keeper cost for {name}: ${kc.cost:.0f} ({effective_source}, {years}yr)")
            case Err(msg):
                typer.echo(f"Error: {msg}", err=True)
                raise typer.Exit(code=1)


@keeper_app.command("decisions")
def decisions_cmd(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")],
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    threshold: Annotated[float, typer.Option(help="Minimum surplus for keep recommendation")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor for multi-year surplus")] = 0.85,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show keeper decisions ranked by surplus value."""
    defaults = load_cli_defaults()
    if version is None:
        version = defaults.version
    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return
        valuations = ctx.valuation_repo.get_by_season(season, system, version=version)
        players = ctx.player_repo.all()
        decisions = compute_surplus(keeper_costs, valuations, players, threshold=threshold, decay=decay)

    print_keeper_decisions(decisions)


@keeper_app.command("adjusted-rankings")
def adjusted_rankings_cmd(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name (must match fbm.toml)")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")],
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    threshold: Annotated[float, typer.Option(help="Minimum surplus for keep recommendation")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor for multi-year surplus")] = 0.85,
    top: Annotated[int | None, typer.Option(help="Show only top N players")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show post-keeper adjusted rankings with recalculated replacement levels."""
    defaults = load_cli_defaults()
    if version is None:
        version = defaults.version
    league_settings = load_league(league, Path.cwd())

    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return

        valuations = ctx.valuation_repo.get_by_season(season, system, version=version)
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
        pitcher_projs = [p for p in projections if p.player_type == "pitcher"]
        pitcher_ids = [p.player_id for p in pitcher_projs]
        pitcher_positions = ctx.eligibility_service.get_pitcher_positions(
            season, league_settings, pitcher_ids, projections=pitcher_projs
        )

        results = compute_adjusted_valuations(
            kept_player_ids={(pid, None) for pid in kept_player_ids},
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
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")],
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    decay: Annotated[float, typer.Option(help="Decay factor")] = 0.85,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate a trade using keeper surplus value."""
    defaults = load_cli_defaults()
    if version is None:
        version = defaults.version
    with build_keeper_context(data_dir) as ctx:
        give_ids: list[int] = []
        for name in gives:
            matches = resolve_players(ctx.player_repo, name)
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
            matches = resolve_players(ctx.player_repo, name)
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
        valuations = ctx.valuation_repo.get_by_season(season, system, version=version)
        players = ctx.player_repo.all()

    result = evaluate_trade(give_ids, receive_ids, keeper_costs, valuations, players, decay)
    print_trade_evaluation(result)


def _resolve_player_id(name: str, player_repo: PlayerRepo) -> int:
    """Resolve a player name to a single ID, or exit with an error."""
    matches = resolve_players(player_repo, name)
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
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")],
    max_keepers: Annotated[int, typer.Option(help="Maximum number of keepers")],
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    max_per_position: Annotated[list[str] | None, typer.Option(help="Position limits, e.g. 'c=1'")] = None,
    max_cost: Annotated[float | None, typer.Option(help="Maximum total keeper cost")] = None,
    required: Annotated[list[str] | None, typer.Option(help="Player names that must be kept")] = None,
    league_keepers: Annotated[
        Path | None, typer.Option(help="CSV of other teams' keepers for pool-adjusted mode")
    ] = None,
    round_escalation: Annotated[int, typer.Option(help="Rounds earlier keeper costs next year")] = 0,
    max_per_round: Annotated[int | None, typer.Option(help="Max keepers from the same effective round")] = None,
    protected_rounds: Annotated[
        list[int] | None, typer.Option(help="Rounds that can't be used as keeper slots")
    ] = None,
    threshold: Annotated[float, typer.Option(help="Minimum surplus for candidates")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor")] = 0.85,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Find the optimal keeper set maximizing total surplus."""
    defaults = load_cli_defaults()
    if version is None:
        version = defaults.version
    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return
        valuations = ctx.valuation_repo.get_by_season(season, system, version=version)
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
            round_escalation=round_escalation,
            max_per_round=max_per_round,
            protected_rounds=frozenset(protected_rounds) if protected_rounds else None,
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
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")],
    max_keepers: Annotated[int, typer.Option(help="Maximum number of keepers")],
    scenario: Annotated[list[str], typer.Option(help="Scenario as 'Name:Player1,Player2'")],
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    threshold: Annotated[float, typer.Option(help="Minimum surplus for candidates")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor")] = 0.85,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Compare named keeper scenarios side-by-side."""
    defaults = load_cli_defaults()
    if version is None:
        version = defaults.version
    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return
        valuations = ctx.valuation_repo.get_by_season(season, system, version=version)
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
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")],
    max_keepers: Annotated[int, typer.Option(help="Maximum number of keepers")],
    acquire: Annotated[list[str] | None, typer.Option(help="Player names to acquire")] = None,
    release: Annotated[list[str] | None, typer.Option(help="Player names to release")] = None,
    acquire_cost: Annotated[list[float] | None, typer.Option(help="Keeper cost for each acquired player")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    threshold: Annotated[float, typer.Option(help="Minimum surplus for candidates")] = 0.0,
    decay: Annotated[float, typer.Option(help="Decay factor")] = 0.85,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show how acquiring/releasing players changes the optimal keeper set."""
    defaults = load_cli_defaults()
    if version is None:
        version = defaults.version
    with build_keeper_context(data_dir) as ctx:
        keeper_costs = ctx.keeper_repo.find_by_season_league(season, league)
        if not keeper_costs:
            typer.echo("No keeper costs found for the specified season and league.")
            return
        valuations = ctx.valuation_repo.get_by_season(season, system, version=version)
        players = ctx.player_repo.all()
        candidates = compute_surplus(keeper_costs, valuations, players, threshold=threshold, decay=decay)

        # Build valuation lookup keyed by (player_id, player_type) to avoid
        # collisions for two-way players.
        val_lookup: dict[tuple[int, PlayerType], float] = {}
        val_pos: dict[tuple[int, PlayerType], str] = {}
        for v in valuations:
            key = (v.player_id, v.player_type)
            if key not in val_lookup or v.value > val_lookup[key]:
                val_lookup[key] = v.value
                val_pos[key] = v.position

        acquire_decisions: list[KeeperDecision] = []
        if acquire:
            costs = acquire_cost or [0.0] * len(acquire)
            for i, name in enumerate(acquire):
                pid = _resolve_player_id(name, ctx.player_repo)
                cost = costs[i] if i < len(costs) else 0.0
                # Find best value across all player types for this player_id
                value = 0.0
                pos = "util"
                best_ptype = PlayerType.BATTER
                for (vid, _vtype), v in val_lookup.items():
                    if vid == pid and v > value:
                        value = v
                        pos = val_pos[(vid, _vtype)]
                        best_ptype = _vtype
                player = ctx.player_repo.get_by_id(pid)
                pname = f"{player.name_first} {player.name_last}" if player else name
                acquire_decisions.append(
                    KeeperDecision(
                        player_id=pid,
                        player_name=pname,
                        player_type=best_ptype,
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


@keeper_app.command("league-set")
def league_set_cmd(
    player_name: Annotated[str, typer.Argument(help="Player name to search for")],
    team: Annotated[str, typer.Option("--team", help="Team name that is keeping the player")],
    season: Annotated[int | None, typer.Option("--season", help="Season year")] = None,
    league: Annotated[str, typer.Option(help="League name")] = "dynasty",
    cost: Annotated[float | None, typer.Option(help="Keeper cost")] = None,
    player_type: Annotated[str | None, typer.Option("--player-type", help="Player type (batter/pitcher)")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Set a single player as kept by a specific team."""
    defaults = load_cli_defaults()
    if season is None:
        season = defaults.season

    with build_keeper_context(data_dir) as ctx:
        matches = resolve_players(ctx.player_repo, player_name)
        if len(matches) == 0:
            typer.echo(f"Error: no player found matching '{player_name}'", err=True)
            raise typer.Exit(code=1)
        if len(matches) > 1:
            names = [f"{p.name_first} {p.name_last}" for p in matches]
            typer.echo(f"Error: ambiguous name '{player_name}', matches: {', '.join(names)}", err=True)
            raise typer.Exit(code=1)
        player = matches[0]
        assert player.id is not None  # noqa: S101

        keeper = LeagueKeeper(
            player_id=player.id,
            season=season,
            league=league,
            team_name=team,
            cost=cost,
            player_type=PlayerType(player_type) if player_type else None,
        )
        ctx.league_keeper_repo.upsert_batch([keeper])
        ctx.conn.commit()

    name = f"{player.name_first} {player.name_last}"
    cost_str = f" (${cost:.0f})" if cost is not None else ""
    type_str = f" [{player_type}]" if player_type is not None else ""
    typer.echo(f"Set {name} as keeper for {team}{cost_str}{type_str}")


@keeper_app.command("league-import")
def league_import_cmd(
    csv_path: Annotated[Path, typer.Argument(help="Path to league keepers CSV file")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Import league-wide keepers from a CSV file."""
    if not csv_path.exists():
        typer.echo(f"Error: file not found: {csv_path}", err=True)
        raise typer.Exit(code=1)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with build_keeper_context(data_dir) as ctx:
        players = ctx.player_repo.all()
        result = import_league_keepers(rows, ctx.league_keeper_repo, players, season=season, league=league)
        ctx.conn.commit()

    typer.echo(f"Loaded {result.loaded} league keepers, skipped {result.skipped}")
    if result.unmatched:
        typer.echo(f"Unmatched players ({len(result.unmatched)}): {', '.join(result.unmatched)}")


@keeper_app.command("league-list")
def league_list_cmd(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show the current league keeper list for a season/league."""
    with build_keeper_context(data_dir) as ctx:
        keepers = ctx.league_keeper_repo.find_by_season_league(season, league)
        players = ctx.player_repo.all()

    print_league_keepers(keepers, players)


@keeper_app.command("league-clear")
def league_clear_cmd(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league: Annotated[str, typer.Option(help="League name")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Remove all league keepers for a season/league."""
    with build_keeper_context(data_dir) as ctx:
        count = ctx.league_keeper_repo.delete_by_season_league(season, league)
        ctx.conn.commit()

    typer.echo(f"Removed {count} league keeper(s)")
