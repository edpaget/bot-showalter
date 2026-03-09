from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._defaults import _DataDirOpt  # noqa: TC001 — used at runtime by typer
from fantasy_baseball_manager.cli._helpers import parse_system_version
from fantasy_baseball_manager.cli._output import (
    print_cohort_bias_report,
    print_cohort_bias_summary,
    print_error,
    print_error_decomposition_report,
    print_feature_gap_report,
)
from fantasy_baseball_manager.cli.factory import build_residuals_context
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.models.statcast_gbm.features import batter_feature_columns, pitcher_feature_columns

residuals_app = typer.Typer(name="residuals", help="Residual analysis tools")


@residuals_app.command("worst-misses")
def worst_misses(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    target: Annotated[str, typer.Option("--target", help="Target stat (e.g. slg, era)")],
    top: Annotated[int, typer.Option("--top", help="Number of worst misses")] = 20,
    direction: Annotated[str | None, typer.Option("--direction", help="over or under")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show the worst prediction misses for a given stat."""
    sys_name, version = parse_system_version(system)

    if direction is not None and direction not in ("over", "under"):
        print_error(f"invalid direction '{direction}', expected 'over' or 'under'")
        raise typer.Exit(code=1)

    with build_residuals_context(data_dir) as ctx:
        report = ctx.analyzer.analyze(
            sys_name,
            version,
            season,
            target,
            player_type,
            top_n=top,
            direction=direction,
        )

    if not report.top_misses:
        print_error("no matching projections/actuals found")
        raise typer.Exit(code=1)

    print_error_decomposition_report(report)


@residuals_app.command("gaps")
def gaps(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    target: Annotated[str, typer.Option("--target", help="Target stat (e.g. slg, era)")],
    miss_percentile: Annotated[
        float, typer.Option("--miss-percentile", help="Percentile threshold for poorly-predicted")
    ] = 80.0,
    include_raw: Annotated[bool, typer.Option("--include-raw", help="Include raw statcast columns")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Detect feature distribution gaps between well-predicted and poorly-predicted players."""
    sys_name, version = parse_system_version(system)

    if player_type == "pitcher":
        model_feature_names = frozenset(pitcher_feature_columns())
    else:
        model_feature_names = frozenset(batter_feature_columns())

    extra_features: dict[int, dict[str, float]] | None = None
    if include_raw:
        extra_features = _load_raw_features(data_dir, season, player_type)

    with build_residuals_context(data_dir) as ctx:
        report = ctx.analyzer.detect_feature_gaps(
            sys_name,
            version,
            season,
            target,
            player_type,
            model_feature_names=model_feature_names,
            miss_percentile=miss_percentile,
            extra_features=extra_features,
        )

    if not report.gaps:
        print_error("no feature gaps found (insufficient data or no residuals)")
        raise typer.Exit(code=1)

    print_feature_gap_report(report)


_VALID_DIMENSIONS = ("age", "position", "handedness", "experience")


@residuals_app.command("cohort")
def cohort(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    target: Annotated[str, typer.Option("--target", help="Target stat (e.g. slg, era)")],
    dimension: Annotated[
        str | None, typer.Option("--dimension", help="age, position, handedness, or experience")
    ] = None,
    all_dimensions: Annotated[bool, typer.Option("--all-dimensions", help="Run all four dimensions")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Report systematic bias by demographic cohort."""
    sys_name, version = parse_system_version(system)

    if not all_dimensions and dimension is None:
        print_error("specify --dimension or --all-dimensions")
        raise typer.Exit(code=1)

    if dimension is not None and dimension not in _VALID_DIMENSIONS:
        print_error(f"invalid dimension '{dimension}', expected one of {_VALID_DIMENSIONS}")
        raise typer.Exit(code=1)

    with build_residuals_context(data_dir) as ctx:
        if all_dimensions:
            reports = ctx.analyzer.bias_by_cohort_all_dimensions(sys_name, version, season, target, player_type)
            if all(not r.cohorts for r in reports):
                print_error("no residuals found")
                raise typer.Exit(code=1)
            print_cohort_bias_summary(reports)
        else:
            assert dimension is not None  # noqa: S101
            report = ctx.analyzer.bias_by_cohort(sys_name, version, season, target, player_type, dimension)
            if not report.cohorts:
                print_error("no residuals found")
                raise typer.Exit(code=1)
            print_cohort_bias_report(report)


def _load_raw_features(data_dir: str, season: int, player_type: str) -> dict[int, dict[str, float]]:
    """Load raw statcast features aggregated per player for the given season."""
    statcast_conn = create_statcast_connection(Path(data_dir) / "statcast.db")
    try:
        query = """
            SELECT
                batter AS player_id,
                AVG(release_speed) AS release_speed,
                AVG(release_spin_rate) AS release_spin_rate,
                AVG(launch_speed) AS launch_speed,
                AVG(launch_angle) AS launch_angle,
                AVG(hit_distance_sc) AS hit_distance_sc,
                AVG(CAST(barrel AS REAL)) AS barrel_rate,
                AVG(estimated_ba_using_speedangle) AS xba,
                AVG(estimated_woba_using_speedangle) AS xwoba,
                AVG(estimated_slg_using_speedangle) AS xslg,
                AVG(release_extension) AS release_extension
            FROM statcast_pitch
            WHERE game_year = ?
            GROUP BY batter
            HAVING COUNT(*) >= 50
        """
        if player_type == "pitcher":
            query = query.replace("batter AS player_id", "pitcher AS player_id").replace(
                "GROUP BY batter", "GROUP BY pitcher"
            )

        cursor = statcast_conn.execute(query, (season,))
        col_names = [desc[0] for desc in cursor.description or []]
        rows = cursor.fetchall()

        result: dict[int, dict[str, float]] = {}
        for row in rows:
            player_id = int(row[0])
            features: dict[str, float] = {}
            for i, col in enumerate(col_names[1:], start=1):
                if row[i] is not None:
                    features[col] = float(row[i])
            result[player_id] = features
        return result
    finally:
        statcast_conn.close()
