"""CLI commands for model validation gate."""

from pathlib import Path
from typing import Annotated, Any

import typer

from fantasy_baseball_manager.cli._helpers import parse_params
from fantasy_baseball_manager.cli._output import console, print_error, print_preflight_result
from fantasy_baseball_manager.cli.factory import create_model
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.domain import Err
from fantasy_baseball_manager.features import SqliteDatasetAssembler
from fantasy_baseball_manager.models.statcast_gbm.model import _StatcastGBMBase
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS
from fantasy_baseball_manager.services import preflight_check, score_cv_folds

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
