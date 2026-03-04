"""CLI command for single-target quick evaluation."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from fantasy_baseball_manager.cli._output import console, print_error, print_quick_eval_result
from fantasy_baseball_manager.cli.factory import create_model
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.domain import Err, Experiment, TargetResult
from fantasy_baseball_manager.features import SqliteDatasetAssembler
from fantasy_baseball_manager.models.statcast_gbm.model import _StatcastGBMBase
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS
from fantasy_baseball_manager.repos import SqliteExperimentRepo, SqliteFeatureCandidateRepo
from fantasy_baseball_manager.services import (
    candidate_values_to_dict,
    inject_candidate_values,
    quick_eval,
    remap_candidate_keys,
    resolve_feature,
)


def _parse_params(raw_params: list[str] | None) -> dict[str, Any] | None:
    if not raw_params:
        return None
    parsed: dict[str, Any] = {}
    for param in raw_params:
        key, _, value = param.partition("=")
        parsed[key] = _coerce_value(value)
    return parsed


def _coerce_value(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def quick_eval_cmd(  # pragma: no cover
    model: Annotated[str, typer.Argument(help="Name of the projection model")],
    target: Annotated[str, typer.Option("--target", help="Target stat to evaluate (e.g. slg, era)")],
    season: Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")] = None,
    columns: Annotated[list[str] | None, typer.Option("--columns", help="Feature columns (replaces defaults)")] = None,
    inject: Annotated[list[str] | None, typer.Option("--inject", help="Column(s) to inject into feature set")] = None,
    baseline: Annotated[float | None, typer.Option("--baseline", help="Baseline RMSE for delta comparison")] = None,
    param: Annotated[list[str] | None, typer.Option("--param", help="Model param as key=value (repeatable)")] = None,
    experiment: Annotated[
        str | None, typer.Option("--experiment", help="Hypothesis — auto-log result to experiment journal")
    ] = None,
    tags: Annotated[str | None, typer.Option("--tags", help="Comma-separated tags for experiment log")] = None,
    parent_id: Annotated[int | None, typer.Option("--parent-id", help="Parent experiment ID")] = None,
    data_dir: Annotated[str | None, typer.Option("--data-dir", help="Data directory")] = None,
) -> None:
    """Train a single target and evaluate on one holdout season."""
    params = _parse_params(param)
    config = load_config(model_name=model, seasons=season, model_params=params)
    resolved_data_dir = data_dir if data_dir is not None else config.data_dir

    conn = create_connection(Path(resolved_data_dir) / "fbm.db")
    try:
        assembler = SqliteDatasetAssembler(conn, statcast_path=Path(resolved_data_dir) / "statcast.db")
        model_result = create_model(model, assembler=assembler)
        match model_result:
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)

        model_instance = model_result.value

        if not isinstance(model_instance, _StatcastGBMBase):
            print_error(f"Model '{model}' does not support quick-eval (not a StatcastGBM model)")
            raise typer.Exit(code=1)

        # Determine player type from target
        is_batter = target in BATTER_TARGETS
        is_pitcher = target in PITCHER_TARGETS
        if not is_batter and not is_pitcher:
            print_error(
                f"Unknown target '{target}'. Valid batter targets: {', '.join(BATTER_TARGETS)}. "
                f"Valid pitcher targets: {', '.join(PITCHER_TARGETS)}."
            )
            raise typer.Exit(code=1)

        # Resolve feature columns
        if columns is not None:
            feature_columns = list(columns)
        elif is_batter:
            feature_columns = list(model_instance._batter_columns)
        else:
            feature_columns = list(model_instance._pitcher_columns)

        if inject:
            for col in inject:
                if col not in feature_columns:
                    feature_columns.append(col)

        # Materialize data
        if is_batter:
            fs = model_instance._batter_training_set_builder(config.seasons)
        else:
            fs = model_instance._pitcher_training_set_builder(config.seasons)

        handle = assembler.get_or_materialize(fs)
        all_rows = assembler.read(handle)

        # Group rows by season
        rows_by_season: dict[int, list[dict[str, Any]]] = {}
        for row in all_rows:
            s = row["season"]
            rows_by_season.setdefault(s, []).append(row)

        # Resolve injected columns that are missing from materialized rows
        if inject:
            first_rows = next(iter(rows_by_season.values()), [])
            existing_keys = first_rows[0].keys() if first_rows else set()
            missing_injected = [c for c in inject if c not in existing_keys]
            if missing_injected:
                player_type = "batter" if is_batter else "pitcher"
                statcast_conn = create_statcast_connection(Path(resolved_data_dir) / "statcast.db")
                try:
                    candidate_repo = SqliteFeatureCandidateRepo(conn)
                    all_seasons = list(rows_by_season.keys())
                    mlbam_to_internal: dict[int, int] = dict(conn.execute("SELECT mlbam_id, id FROM player").fetchall())
                    for col_name in missing_injected:
                        cv = resolve_feature(col_name, statcast_conn, candidate_repo, all_seasons, player_type)
                        values_dict = candidate_values_to_dict(cv)
                        remapped = remap_candidate_keys(values_dict, mlbam_to_internal)
                        inject_candidate_values(rows_by_season, col_name, remapped)
                finally:
                    statcast_conn.close()

        # Split: train on all but last season, holdout on last
        sorted_seasons = sorted(rows_by_season.keys())
        if len(sorted_seasons) < 2:
            print_error(f"Need at least 2 seasons for train/holdout split (got {len(sorted_seasons)})")
            raise typer.Exit(code=1)

        train_seasons = sorted_seasons[:-1]
        holdout_season = sorted_seasons[-1]

        console.print(f"Quick-eval [bold]{target}[/bold] on model [bold]{model}[/bold]")
        console.print(f"  Train: {train_seasons}  Holdout: {holdout_season}")
        console.print(f"  Features: {len(feature_columns)} columns")

        eval_result = quick_eval(
            feature_columns=feature_columns,
            target=target,
            rows_by_season=rows_by_season,
            train_seasons=train_seasons,
            holdout_season=holdout_season,
            params=params,
            baseline_rmse=baseline,
        )

        print_quick_eval_result(eval_result)

        if experiment is not None:
            if eval_result.baseline_rmse is None:
                print_error("--baseline is required when using --experiment")
                raise typer.Exit(code=1)

            target_result = TargetResult(
                rmse=eval_result.rmse,
                baseline_rmse=eval_result.baseline_rmse,
                delta=eval_result.delta,  # type: ignore[arg-type]
                delta_pct=eval_result.delta_pct,  # type: ignore[arg-type]
            )

            injected = list(inject) if inject else []
            feature_diff = {"added": injected, "removed": []}

            parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

            direction = "improved" if eval_result.delta < 0 else "worsened"  # type: ignore[operator]
            conclusion = (
                f"{target} {direction} by {abs(eval_result.delta_pct):.2f}% "  # type: ignore[arg-type]
                f"(RMSE {eval_result.baseline_rmse:.4f} → {eval_result.rmse:.4f})"
            )

            exp = Experiment(
                timestamp=datetime.now(UTC).isoformat(),
                hypothesis=experiment,
                model=model,
                player_type="batter" if is_batter else "pitcher",
                feature_diff=feature_diff,
                seasons={"train": train_seasons, "holdout": [holdout_season]},
                params=params or {},
                target_results={target: target_result},
                conclusion=conclusion,
                tags=parsed_tags,
                parent_id=parent_id,
            )

            repo = SqliteExperimentRepo(conn)
            exp_id = repo.save(exp)
            conn.commit()
            console.print(f"Logged experiment [bold green]#{exp_id}[/bold green]")
    finally:
        conn.close()
