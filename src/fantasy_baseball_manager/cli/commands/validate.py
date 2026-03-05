"""CLI commands for model validation gate."""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from fantasy_baseball_manager.cli._helpers import parse_params
from fantasy_baseball_manager.cli._output import console, print_error, print_preflight_result, print_validation_result
from fantasy_baseball_manager.cli.factory import build_eval_context, build_model_context, create_model
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain import Err, ModelConfig
from fantasy_baseball_manager.features import SqliteDatasetAssembler
from fantasy_baseball_manager.models.statcast_gbm.model import _StatcastGBMBase
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS
from fantasy_baseball_manager.services import (
    FullValidationConfig,
    FullValidationRunner,
    preflight_check,
    score_cv_folds,
)

validate_app = typer.Typer(name="validate", help="Validation gate commands")


def preflight_cmd(
    model: Annotated[str, typer.Argument(help="Name of the projection model")],
    candidate_columns: Annotated[
        str, typer.Option("--candidate-columns", help="Comma-separated candidate feature columns")
    ],
    player_type: Annotated[str, typer.Option("--player-type", help="Player type: 'batter' or 'pitcher'")],
    season: Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")] = None,
    param: Annotated[list[str] | None, typer.Option("--param", help="Model param as key=value")] = None,
    data_dir: Annotated[str | None, typer.Option("--data-dir", help="Data directory")] = None,
) -> None:
    """Run pre-flight confidence check comparing candidate vs baseline features."""
    if player_type not in ("batter", "pitcher"):
        print_error(f"Invalid player type '{player_type}'. Must be 'batter' or 'pitcher'.")
        raise typer.Exit(code=1)

    params = parse_params(param)
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
            print_error(f"Model '{model}' does not support validate (not a StatcastGBM model)")
            raise typer.Exit(code=1)

        is_batter = player_type == "batter"

        # Resolve baseline columns (model defaults)
        baseline_columns = list(model_instance._batter_columns if is_batter else model_instance._pitcher_columns)

        # Resolve candidate columns (baseline + candidate additions)
        candidate_cols = [c.strip() for c in candidate_columns.split(",")]
        all_candidate_columns = list(baseline_columns)
        for col in candidate_cols:
            if col not in all_candidate_columns:
                all_candidate_columns.append(col)

        # Resolve targets
        targets: list[str] = list(BATTER_TARGETS if is_batter else PITCHER_TARGETS)

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

        sorted_seasons = sorted(rows_by_season.keys())
        if len(sorted_seasons) < 3:
            print_error(f"Need at least 3 seasons for pre-flight CV (got {len(sorted_seasons)})")
            raise typer.Exit(code=1)

        console.print(f"Pre-flight check on model [bold]{model}[/bold] ({player_type})")
        console.print(f"  Seasons: {sorted_seasons}")
        console.print(f"  Baseline: {len(baseline_columns)} columns")
        console.print(
            f"  Candidate: {len(all_candidate_columns)} columns (+{len(all_candidate_columns) - len(baseline_columns)})"
        )

        gbm_params: dict[str, int | float] = params if params is not None else {}

        baseline_cv = score_cv_folds(baseline_columns, targets, rows_by_season, sorted_seasons, gbm_params)
        candidate_cv = score_cv_folds(all_candidate_columns, targets, rows_by_season, sorted_seasons, gbm_params)

        result = preflight_check(candidate_cv, baseline_cv)
        print_preflight_result(result)
    finally:
        conn.close()


validate_app.command("preflight")(preflight_cmd)


def full_cmd(
    model: Annotated[str, typer.Argument(help="Name of the projection model")],
    old_version: Annotated[str, typer.Option("--old-version", help="Baseline model version")],
    new_version: Annotated[str, typer.Option("--new-version", help="Candidate model version")],
    holdout: Annotated[list[int], typer.Option("--holdout", help="Holdout season(s)")],
    train: Annotated[list[int], typer.Option("--train", help="Training season(s)")],
    new_params: Annotated[str | None, typer.Option("--new-params", help="JSON params for new version")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Top-N player check")] = None,
    param: Annotated[list[str] | None, typer.Option("--param", help="Shared model param as key=value")] = None,
    candidate_columns: Annotated[
        str | None, typer.Option("--candidate-columns", help="Comma-separated candidate feature columns")
    ] = None,
    player_type: Annotated[
        str | None, typer.Option("--player-type", help="Player type for pre-flight: 'batter' or 'pitcher'")
    ] = None,
    keep: Annotated[bool, typer.Option("--keep", help="Keep gate predictions after run")] = False,
    data_dir: Annotated[str | None, typer.Option("--data-dir", help="Data directory")] = None,
) -> None:
    """Run full validation: train, predict, and compare on multiple holdout seasons."""
    old_params_dict: dict[str, Any] = parse_params(param) or {}
    new_params_dict: dict[str, Any] = dict(old_params_dict)
    if new_params is not None:
        new_params_dict.update(json.loads(new_params))

    config = load_config(model_name=model, seasons=train)
    resolved_data_dir = data_dir if data_dir is not None else config.data_dir

    model_config = ModelConfig(
        data_dir=resolved_data_dir,
        artifacts_dir=config.artifacts_dir,
        seasons=train,
        model_params=old_params_dict,
    )

    with build_model_context(model, model_config) as model_ctx, build_eval_context(resolved_data_dir) as eval_ctx:
        # Optional pre-flight
        preflight_result = None
        if candidate_columns is not None and player_type is not None:
            if not isinstance(model_ctx.model, _StatcastGBMBase):
                console.print("[yellow]Pre-flight skipped: model is not a StatcastGBM model[/yellow]")
            else:
                preflight_result = _run_preflight(
                    model_instance=model_ctx.model,
                    player_type=player_type,
                    candidate_columns=candidate_columns,
                    seasons=train,
                    params=old_params_dict,
                    data_dir=resolved_data_dir,
                    conn=model_ctx.conn,
                )

        validation_config = FullValidationConfig(
            model_name=model,
            old_version=old_version,
            new_version=new_version,
            old_params=old_params_dict,
            new_params=new_params_dict,
            holdout_seasons=holdout,
            train_seasons=train,
            top=top,
            data_dir=resolved_data_dir,
            artifacts_dir=config.artifacts_dir,
        )

        runner = FullValidationRunner(
            model=model_ctx.model,
            evaluator=eval_ctx.evaluator,
            projection_repo=eval_ctx.projection_repo,
        )

        result = runner.run(validation_config, preflight=preflight_result)
        print_validation_result(result)

        if not keep:
            runner.cleanup(validation_config)

        if not result.passed:
            raise typer.Exit(code=1)


def _run_preflight(
    *,
    model_instance: _StatcastGBMBase,
    player_type: str,
    candidate_columns: str,
    seasons: list[int],
    params: dict[str, Any],
    data_dir: str,
    conn: Any,
) -> Any:
    """Run pre-flight check and return PreflightResult, or None on error."""
    is_batter = player_type == "batter"

    baseline_columns = list(model_instance._batter_columns if is_batter else model_instance._pitcher_columns)

    candidate_cols = [c.strip() for c in candidate_columns.split(",")]
    all_candidate_columns = list(baseline_columns)
    for col in candidate_cols:
        if col not in all_candidate_columns:
            all_candidate_columns.append(col)

    targets: list[str] = list(BATTER_TARGETS if is_batter else PITCHER_TARGETS)

    assembler = SqliteDatasetAssembler(conn, statcast_path=Path(data_dir) / "statcast.db")
    if is_batter:
        fs = model_instance._batter_training_set_builder(seasons)
    else:
        fs = model_instance._pitcher_training_set_builder(seasons)

    handle = assembler.get_or_materialize(fs)
    all_rows = assembler.read(handle)

    rows_by_season: dict[int, list[dict[str, Any]]] = {}
    for row in all_rows:
        s = row["season"]
        rows_by_season.setdefault(s, []).append(row)

    sorted_seasons = sorted(rows_by_season.keys())
    if len(sorted_seasons) < 3:
        console.print(f"[yellow]Pre-flight skipped: need ≥3 seasons (got {len(sorted_seasons)})[/yellow]")
        return None

    gbm_params: dict[str, int | float] = params if params else {}

    baseline_cv = score_cv_folds(baseline_columns, targets, rows_by_season, sorted_seasons, gbm_params)
    candidate_cv = score_cv_folds(all_candidate_columns, targets, rows_by_season, sorted_seasons, gbm_params)

    pf_result = preflight_check(candidate_cv, baseline_cv)
    print_preflight_result(pf_result)
    return pf_result


validate_app.command("full")(full_cmd)
