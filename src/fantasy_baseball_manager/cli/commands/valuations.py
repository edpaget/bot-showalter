from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import click
import typer

from fantasy_baseball_manager.cli._defaults import _DataDirOpt, load_cli_defaults
from fantasy_baseball_manager.cli._output import (
    print_player_valuations,
    print_valuation_comparison,
    print_valuation_eval_result,
    print_valuation_rankings,
    print_valuation_regression_check,
)
from fantasy_baseball_manager.cli.factory import (
    build_sgp_context,
    build_valuation_eval_context,
    build_valuations_context,
)
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain import check_valuation_regression
from fantasy_baseball_manager.services import compute_sgp_denominators, find_league_lineage

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import CategoryConfig, SgpDenominators

valuations_app = typer.Typer(name="valuations", help="Look up and explore player valuations")

_DEFAULT_MIN_VALUE = 0.01


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


def _parse_targets(targets_opt: str | None) -> frozenset[str] | None:
    valid_targets = {"war", "hit-rate"}
    if targets_opt is None:
        return None
    parsed = frozenset(t.strip() for t in targets_opt.split(","))
    invalid = parsed - valid_targets
    if invalid:
        raise typer.BadParameter(
            f"Unknown targets: {', '.join(sorted(invalid))}. Valid: {', '.join(sorted(valid_targets))}"
        )
    return parsed


def _resolve_min_value(min_value: float | None, full: bool, *, min_value_given: bool) -> float | None:
    """Resolve the effective min_value for evaluate/compare commands.

    Priority: explicit --min-value > --full > default (0.01).
    """
    if min_value_given:
        return min_value
    if full:
        return None
    return _DEFAULT_MIN_VALUE


@valuations_app.command("evaluate")
def valuations_evaluate(
    ctx: typer.Context,
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = None,
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N mispricings")] = None,
    min_value: Annotated[
        float | None, typer.Option("--min-value", help="Min predicted or actual value to include")
    ] = None,
    top_n: Annotated[int | None, typer.Option("--top-n", help="Top N by predicted rank for population filter")] = None,
    targets_opt: Annotated[
        str | None, typer.Option("--targets", help="Comma-separated targets: war,hit-rate (default: all)")
    ] = None,
    stratify: Annotated[str | None, typer.Option("--stratify", help="Stratify by: player_type")] = None,
    tail: Annotated[bool, typer.Option("--tail", help="Include top-25 and top-50 tail accuracy")] = False,
    full: Annotated[bool, typer.Option("--full", help="Include all players (no min-value filter)")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate valuation accuracy against end-of-season actuals."""
    defaults = load_cli_defaults()
    if system is None:
        system = defaults.system
    if version is None:
        version = defaults.version

    min_value_given = ctx.get_parameter_source("min_value") == click.core.ParameterSource.COMMANDLINE
    effective_min_value = _resolve_min_value(min_value, full, min_value_given=min_value_given)
    targets = _parse_targets(targets_opt)

    valid_stratify = {"player_type"}
    if stratify is not None and stratify not in valid_stratify:
        raise typer.BadParameter(f"Unknown stratify value: {stratify}. Valid: {', '.join(sorted(valid_stratify))}")

    tail_ns: tuple[int, ...] | None = (25, 50) if tail else None

    league = load_league(league_name, Path.cwd())
    with build_valuation_eval_context(data_dir) as eval_ctx:
        result = eval_ctx.evaluator.evaluate(
            system,
            version,
            season,
            league,
            top=top_n,
            min_value=effective_min_value,
            targets=targets,
            stratify=stratify,
            tail_ns=tail_ns,
        )
    print_valuation_eval_result(result, top=top)


@valuations_app.command("compare")
def valuations_compare(
    systems: Annotated[list[str], typer.Argument(help="Two system/version pairs (e.g. zar/holdout zar-v2/holdout)")],
    ctx: typer.Context,
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    min_value: Annotated[
        float | None, typer.Option("--min-value", help="Min predicted or actual value to include")
    ] = None,
    top_n: Annotated[int | None, typer.Option("--top-n", help="Top N by predicted rank for population filter")] = None,
    targets_opt: Annotated[
        str | None, typer.Option("--targets", help="Comma-separated targets: war,hit-rate (default: all)")
    ] = None,
    stratify: Annotated[str | None, typer.Option("--stratify", help="Stratify by: player_type")] = None,
    tail: Annotated[bool, typer.Option("--tail", help="Include top-25 and top-50 tail accuracy")] = False,
    check: Annotated[bool, typer.Option("--check", help="Exit non-zero on regression")] = False,
    full: Annotated[bool, typer.Option("--full", help="Include all players (no min-value filter)")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Compare two valuation systems side-by-side."""
    if len(systems) != 2:
        raise typer.BadParameter("Exactly 2 system/version pairs required (e.g. zar/holdout zar-v2/holdout)")

    def _parse_system_version(spec: str) -> tuple[str, str]:
        parts = spec.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise typer.BadParameter(f"Invalid system/version format: '{spec}'. Expected 'system/version'.")
        return parts[0], parts[1]

    baseline_system, baseline_version = _parse_system_version(systems[0])
    candidate_system, candidate_version = _parse_system_version(systems[1])

    min_value_given = ctx.get_parameter_source("min_value") == click.core.ParameterSource.COMMANDLINE
    effective_min_value = _resolve_min_value(min_value, full, min_value_given=min_value_given)
    targets = _parse_targets(targets_opt)

    valid_stratify = {"player_type"}
    if stratify is not None and stratify not in valid_stratify:
        raise typer.BadParameter(f"Unknown stratify value: {stratify}. Valid: {', '.join(sorted(valid_stratify))}")

    tail_ns: tuple[int, ...] | None = (25, 50) if tail else None

    league = load_league(league_name, Path.cwd())
    with build_valuation_eval_context(data_dir) as eval_ctx:
        comparison = eval_ctx.evaluator.compare(
            baseline_system,
            baseline_version,
            candidate_system,
            candidate_version,
            season,
            league,
            min_value=effective_min_value,
            top=top_n,
            targets=targets,
            stratify=stratify,
            tail_ns=tail_ns,
        )

    print_valuation_comparison(comparison)

    if check:
        regression = check_valuation_regression(comparison.baseline, comparison.candidate)
        print_valuation_regression_check(regression)
        if not regression.passed:
            raise SystemExit(1)


@valuations_app.command("sgp-denominators")
def sgp_denominators(
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "keeper",
    seasons: Annotated[int | None, typer.Option("--seasons", help="Limit to last N seasons")] = None,
    yahoo_league: Annotated[str | None, typer.Option("--yahoo-league", help="Starting Yahoo league key")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Compute and display SGP denominators from league standings."""
    settings = load_league(league_name, Path.cwd())
    all_categories = list(settings.batting_categories) + list(settings.pitching_categories)

    with build_sgp_context(data_dir) as ctx:
        all_leagues = ctx.yahoo_league_repo.get_all()

        if yahoo_league:
            start_key = yahoo_league
        else:
            redraft_leagues = [lg for lg in all_leagues if not lg.is_keeper]
            if not redraft_leagues:
                typer.echo("No redraft leagues found in database.", err=True)
                raise typer.Exit(1)
            redraft_leagues.sort(key=lambda lg: lg.season, reverse=True)
            start_key = redraft_leagues[0].league_key

        lineage_keys = find_league_lineage(all_leagues, start_key)
        if not lineage_keys:
            typer.echo(f"No league lineage found for {start_key}", err=True)
            raise typer.Exit(1)

        all_standings = []
        for league_key in lineage_keys:
            league = next((lg for lg in all_leagues if lg.league_key == league_key), None)
            if league is None:
                continue
            team_stats = ctx.yahoo_team_stats_repo.get_by_league_season(league_key, league.season)
            all_standings.extend(team_stats)

        if seasons is not None:
            all_seasons_list = sorted({ts.season for ts in all_standings})
            keep_seasons = set(all_seasons_list[-seasons:])
            all_standings = [ts for ts in all_standings if ts.season in keep_seasons]

        if not all_standings:
            typer.echo("No standings data found.", err=True)
            raise typer.Exit(1)

        result = compute_sgp_denominators(all_standings, all_categories)

    _print_sgp_denominators(result, all_categories)


def _print_sgp_denominators(result: SgpDenominators, categories: list[CategoryConfig]) -> None:
    season_set = sorted({sd.season for sd in result.per_season})

    header = f"{'Category':<10} {'Avg':>8}"
    for s in season_set:
        header += f" {s:>8}"
    typer.echo(header)
    typer.echo("-" * len(header))

    for cat in categories:
        if cat.key not in result.averages:
            continue
        avg = result.averages[cat.key]
        row = f"{cat.key:<10} {avg:>8.3f}"
        for s in season_set:
            season_val = next(
                (sd.denominator for sd in result.per_season if sd.category == cat.key and sd.season == s),
                None,
            )
            if season_val is not None:
                row += f" {season_val:>8.3f}"
            else:
                row += f" {'—':>8}"
        typer.echo(row)
