"""CLI command for feature set A/B comparison."""

from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import console, print_compare_features_result, print_error
from fantasy_baseball_manager.cli.commands.quick_eval import _parse_params
from fantasy_baseball_manager.cli.factory import DbLabelSource, create_model
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.domain import Err, Experimentable
from fantasy_baseball_manager.features import SqliteDatasetAssembler
from fantasy_baseball_manager.repos import SqliteFeatureCandidateRepo
from fantasy_baseball_manager.services import (
    compare_feature_sets,
    resolve_and_inject_candidates,
)


def compare_features_cmd(  # pragma: no cover
    model: Annotated[str, typer.Argument(help="Name of the projection model")],
    set_a: Annotated[str, typer.Option("--set-a", help="Feature set A: comma-separated columns or 'default'")],
    set_b: Annotated[str, typer.Option("--set-b", help="Feature set B: comma-separated columns")],
    player_type: Annotated[str, typer.Option("--player-type", help="Player type: 'batter' or 'pitcher'")],
    season: Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")] = None,
    param: Annotated[list[str] | None, typer.Option("--param", help="Model param as key=value")] = None,
    data_dir: Annotated[str | None, typer.Option("--data-dir", help="Data directory")] = None,
) -> None:
    """Compare two feature sets on identical data splits."""
    if player_type not in ("batter", "pitcher"):
        print_error(f"Invalid player type '{player_type}'. Must be 'batter' or 'pitcher'.")
        raise typer.Exit(code=1)

    params = _parse_params(param)
    config = load_config(model_name=model, seasons=season, model_params=params)
    resolved_data_dir = data_dir if data_dir is not None else config.data_dir

    conn = create_connection(Path(resolved_data_dir) / "fbm.db")
    try:
        assembler = SqliteDatasetAssembler(
            SingleConnectionProvider(conn), statcast_path=Path(resolved_data_dir) / "statcast.db"
        )
        model_result = create_model(model, assembler=assembler, label_source=DbLabelSource(conn))
        match model_result:
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)

        model_instance = model_result.value

        if not isinstance(model_instance, Experimentable):
            print_error(f"Model '{model}' does not support compare-features (model does not implement Experimentable)")
            raise typer.Exit(code=1)

        # Resolve set A columns
        if set_a == "default":
            columns_a = model_instance.experiment_feature_columns(player_type)
        else:
            columns_a = [c.strip() for c in set_a.split(",")]

        # Resolve set B columns
        columns_b = [c.strip() for c in set_b.split(",")]

        # Resolve targets and training data
        targets = model_instance.experiment_targets(player_type)
        rows_by_season = model_instance.experiment_training_data(player_type, config.seasons)

        # Resolve columns in set A or B that are missing from materialized rows
        first_rows = next(iter(rows_by_season.values()), [])
        existing_keys = first_rows[0].keys() if first_rows else set()
        all_columns = set(columns_a) | set(columns_b)
        missing_columns = [c for c in all_columns if c not in existing_keys]
        if missing_columns:
            lags, weights = model_instance.experiment_candidate_lags(player_type)
            statcast_conn = create_statcast_connection(Path(resolved_data_dir) / "statcast.db")
            try:
                candidate_repo = SqliteFeatureCandidateRepo(SingleConnectionProvider(conn))
                mlbam_to_internal: dict[int, int] = dict(conn.execute("SELECT mlbam_id, id FROM player").fetchall())
                resolve_and_inject_candidates(
                    rows_by_season,
                    missing_columns,
                    statcast_conn,
                    candidate_repo,
                    mlbam_to_internal,
                    player_type,
                    lags,
                    weights,
                )
            finally:
                statcast_conn.close()

        sorted_seasons = sorted(rows_by_season.keys())
        if len(sorted_seasons) < 2:
            print_error(f"Need at least 2 seasons for comparison (got {len(sorted_seasons)})")
            raise typer.Exit(code=1)

        mode = "CV" if len(sorted_seasons) >= 3 else "single-holdout"
        console.print(f"Compare features on model [bold]{model}[/bold] ({player_type}, {mode})")
        console.print(f"  Seasons: {sorted_seasons}")
        console.print(f"  Set A: {len(columns_a)} columns")
        console.print(f"  Set B: {len(columns_b)} columns")

        backend = model_instance.experiment_training_backend()

        result = compare_feature_sets(
            columns_a=columns_a,
            columns_b=columns_b,
            targets=targets,
            rows_by_season=rows_by_season,
            seasons=sorted_seasons,
            params=params,
            backend=backend,
        )

        print_compare_features_result(result)
    finally:
        conn.close()
