from datetime import date
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import (
    print_bin_target_means,
    print_binned_summary,
    print_candidate_values,
    print_correlation_results,
    print_error,
    print_interaction_scan_results,
)
from fantasy_baseball_manager.cli.factory import FeatureContext, build_feature_context
from fantasy_baseball_manager.domain import CandidateValue, CorrelationScanResult, FeatureCandidate, PlayerType
from fantasy_baseball_manager.services import (
    BINNING_METHODS,
    INTERACTION_OPERATIONS,
    aggregate_candidate,
    bin_candidate,
    candidate_values_to_dict,
    cross_bin_candidates,
    interact_candidates,
    rank_columns,
    resolve_feature,
)

feature_app = typer.Typer(name="feature", help="Feature candidate exploration tools")


@feature_app.command("candidate")
def candidate_cmd(
    expression: Annotated[str, typer.Argument(help="SQL aggregation expression (e.g. 'AVG(launch_speed)')")],
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")] = ...,  # type: ignore[assignment]
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")] = ...,  # type: ignore[assignment]
    correlate: Annotated[bool, typer.Option("--correlate", help="Run target correlation scan")] = False,
    name: Annotated[str | None, typer.Option("--name", help="Save candidate with this name")] = None,
    min_pa: Annotated[int | None, typer.Option("--min-pa", help="Minimum plate appearances (batters)")] = None,
    min_ip: Annotated[float | None, typer.Option("--min-ip", help="Minimum innings pitched (pitchers)")] = None,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Aggregate statcast data with a SQL expression and optionally correlate with targets."""
    with build_feature_context(data_dir) as ctx:
        try:
            results = aggregate_candidate(
                ctx.statcast_conn,
                expression,
                season,
                player_type,
                min_pa=min_pa,
                min_ip=min_ip,
            )
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1) from e

        print_candidate_values(results)

        if correlate:
            try:
                scan_result = ctx.scanner.scan_target_correlations(expression, season, player_type)
            except ValueError as e:
                print_error(str(e))
                raise typer.Exit(code=1) from e
            print_correlation_results(scan_result)

        if name:
            candidate = FeatureCandidate(
                name=name,
                expression=expression,
                player_type=PlayerType(player_type),
                min_pa=min_pa,
                min_ip=min_ip,
                created_at=date.today().isoformat(),
            )
            ctx.candidate_repo.save(candidate)
            typer.echo(f"Saved candidate '{name}'")


@feature_app.command("interact")
def interact_cmd(
    feature_a: Annotated[str, typer.Argument(help="First feature (name or SQL expression)")],
    feature_b: Annotated[str, typer.Argument(help="Second feature (name or SQL expression)")],
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")] = ...,  # type: ignore[assignment]
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")] = ...,  # type: ignore[assignment]
    op: Annotated[str | None, typer.Option("--op", help="Operation: product, ratio, difference, sum")] = None,
    correlate: Annotated[bool, typer.Option("--correlate", help="Run target correlation scan")] = False,
    scan: Annotated[bool, typer.Option("--scan", help="Try all operations and rank by correlation")] = False,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Compute interaction between two features and optionally correlate with targets."""
    with build_feature_context(data_dir) as ctx:
        try:
            values_a = resolve_feature(feature_a, ctx.statcast_conn, ctx.candidate_repo, season, player_type)
            values_b = resolve_feature(feature_b, ctx.statcast_conn, ctx.candidate_repo, season, player_type)
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1) from e

        if scan:
            _run_scan(ctx, values_a, values_b, season, player_type)
            return

        if op is None:
            print_error("Must specify --op or --scan")
            raise typer.Exit(code=1)

        if op not in INTERACTION_OPERATIONS:
            print_error(f"Invalid operation: {op!r}. Must be one of {sorted(INTERACTION_OPERATIONS)}")
            raise typer.Exit(code=1)

        try:
            results = interact_candidates(values_a, values_b, op)
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1) from e

        print_candidate_values(results)

        if correlate:
            cand_dict = candidate_values_to_dict(results)
            label = f"{feature_a} {op} {feature_b}"
            scan_result = ctx.scanner.scan_from_values(label, cand_dict, season, player_type)
            print_correlation_results(scan_result)


@feature_app.command("bin")
def bin_cmd(
    feature: Annotated[str, typer.Argument(help="Feature name or SQL expression")],
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")] = ...,  # type: ignore[assignment]
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")] = ...,  # type: ignore[assignment]
    method: Annotated[str, typer.Option("--method", help="Binning method: quantile, uniform, custom")] = "quantile",
    bins: Annotated[int, typer.Option("--bins", help="Number of bins")] = 4,
    cross: Annotated[str | None, typer.Option("--cross", help="Second feature for cross-product binning")] = None,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Bin a continuous feature into discrete categories and show within-bin target means."""
    if method not in BINNING_METHODS:
        print_error(f"Invalid method: {method!r}. Must be one of {sorted(BINNING_METHODS)}")
        raise typer.Exit(code=1)

    with build_feature_context(data_dir) as ctx:
        try:
            values = resolve_feature(feature, ctx.statcast_conn, ctx.candidate_repo, season, player_type)
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1) from e

        try:
            binned = bin_candidate(values, method, bins)
        except ValueError as e:
            print_error(str(e))
            raise typer.Exit(code=1) from e

        if cross is not None:
            try:
                cross_values = resolve_feature(cross, ctx.statcast_conn, ctx.candidate_repo, season, player_type)
            except ValueError as e:
                print_error(str(e))
                raise typer.Exit(code=1) from e
            cross_binned = bin_candidate(cross_values, method, bins)
            binned = cross_bin_candidates(binned, cross_binned)

        print_binned_summary(binned)

        # Always show target means
        target_means = ctx.scanner.compute_bin_target_means(binned, season, player_type)
        print_bin_target_means(target_means)


def _run_scan(
    ctx: FeatureContext,
    values_a: list[CandidateValue],
    values_b: list[CandidateValue],
    seasons: list[int],
    player_type: str,
) -> None:
    """Try all four interaction operations, correlate each, rank and print."""
    scan_results: list[CorrelationScanResult] = []
    for op in sorted(INTERACTION_OPERATIONS):
        interaction = interact_candidates(values_a, values_b, op)
        cand_dict = candidate_values_to_dict(interaction)
        if not cand_dict:
            continue
        result = ctx.scanner.scan_from_values(op, cand_dict, seasons, player_type)
        scan_results.append(result)

    rankings = rank_columns(scan_results)
    ranked_pairs = [(r.column_spec, r) for r in rankings]
    print_interaction_scan_results(ranked_pairs)

    if rankings:
        best_op = rankings[0].column_spec
        best_result = next(r for r in scan_results if r.column_spec == best_op)
        typer.echo(f"\nBest operation: {best_op}")
        print_correlation_results(best_result)
