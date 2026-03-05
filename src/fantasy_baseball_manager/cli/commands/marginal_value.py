"""CLI command for marginal value estimation."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import console, print_error, print_marginal_value_results
from fantasy_baseball_manager.cli.commands.quick_eval import _parse_params
from fantasy_baseball_manager.cli.factory import create_model
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.db.statcast_connection import create_statcast_connection
from fantasy_baseball_manager.domain import Err, Experiment, Experimentable, TargetResult
from fantasy_baseball_manager.features import SqliteDatasetAssembler
from fantasy_baseball_manager.repos import SqliteExperimentRepo, SqliteFeatureCandidateRepo
from fantasy_baseball_manager.services import (
    MarginalValueResult,
    candidate_values_to_dict,
    inject_candidate_values,
    marginal_value,
    remap_candidate_keys,
    resolve_feature,
)


def marginal_value_cmd(  # pragma: no cover
    model: Annotated[str, typer.Argument(help="Name of the projection model")],
    candidate: Annotated[list[str], typer.Option("--candidate", help="Candidate column(s) to evaluate")],
    player_type: Annotated[str, typer.Option("--player-type", help="Player type: 'batter' or 'pitcher'")],
    season: Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")] = None,
    param: Annotated[list[str] | None, typer.Option("--param", help="Model param as key=value")] = None,
    experiment: Annotated[
        str | None, typer.Option("--experiment", help="Hypothesis — auto-log results to experiment journal")
    ] = None,
    tags: Annotated[str | None, typer.Option("--tags", help="Comma-separated tags for experiment log")] = None,
    parent_id: Annotated[int | None, typer.Option("--parent-id", help="Parent experiment ID")] = None,
    data_dir: Annotated[str | None, typer.Option("--data-dir", help="Data directory")] = None,
) -> None:
    """Estimate the RMSE improvement from adding candidate feature(s)."""
    if player_type not in ("batter", "pitcher"):
        print_error(f"Invalid player type '{player_type}'. Must be 'batter' or 'pitcher'.")
        raise typer.Exit(code=1)

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

        if not isinstance(model_instance, Experimentable):
            print_error(f"Model '{model}' does not support marginal-value (model does not implement Experimentable)")
            raise typer.Exit(code=1)

        # Resolve feature columns, targets, and training data from model
        feature_columns = model_instance.experiment_feature_columns(player_type)
        targets = model_instance.experiment_targets(player_type)
        rows_by_season = model_instance.experiment_training_data(player_type, config.seasons)

        # Resolve candidate columns that are missing from materialized rows
        first_rows = next(iter(rows_by_season.values()), [])
        existing_keys = first_rows[0].keys() if first_rows else set()
        missing_candidates = [c for c in candidate if c not in existing_keys]
        if missing_candidates:
            statcast_conn = create_statcast_connection(Path(resolved_data_dir) / "statcast.db")
            try:
                candidate_repo = SqliteFeatureCandidateRepo(conn)
                all_seasons = list(rows_by_season.keys())
                mlbam_to_internal: dict[int, int] = dict(conn.execute("SELECT mlbam_id, id FROM player").fetchall())
                for cand_name in missing_candidates:
                    cv = resolve_feature(cand_name, statcast_conn, candidate_repo, all_seasons, player_type)
                    values_dict = candidate_values_to_dict(cv)
                    remapped = remap_candidate_keys(values_dict, mlbam_to_internal)
                    inject_candidate_values(rows_by_season, cand_name, remapped)
            finally:
                statcast_conn.close()

        # Split: train on all but last season, holdout on last
        sorted_seasons = sorted(rows_by_season.keys())
        if len(sorted_seasons) < 2:
            print_error(f"Need at least 2 seasons for train/holdout split (got {len(sorted_seasons)})")
            raise typer.Exit(code=1)

        train_seasons = sorted_seasons[:-1]
        holdout_season = sorted_seasons[-1]

        console.print(f"Marginal value on model [bold]{model}[/bold] ({player_type})")
        console.print(f"  Train: {train_seasons}  Holdout: {holdout_season}")
        console.print(f"  Baseline features: {len(feature_columns)} columns")
        console.print(f"  Candidates: {', '.join(candidate)}")

        results: list[MarginalValueResult] = []
        for cand in candidate:
            result = marginal_value(
                candidate_column=cand,
                feature_columns=feature_columns,
                targets=targets,
                rows_by_season=rows_by_season,
                train_seasons=train_seasons,
                holdout_season=holdout_season,
                params=params,
            )
            results.append(result)

        # Sort by avg_delta_pct (ascending — most negative/best first)
        results.sort(key=lambda r: r.avg_delta_pct)

        print_marginal_value_results(results)

        if experiment is not None:
            parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            repo = SqliteExperimentRepo(conn)

            for mv_result in results:
                target_results_dict = {
                    d.target: TargetResult(
                        rmse=d.candidate_rmse,
                        baseline_rmse=d.baseline_rmse,
                        delta=d.delta,
                        delta_pct=d.delta_pct,
                    )
                    for d in mv_result.deltas
                }

                direction = "improved" if mv_result.avg_delta_pct < 0 else "worsened"
                conclusion = (
                    f"{mv_result.candidate} {direction} {mv_result.n_improved}/{mv_result.n_total} targets "
                    f"(avg {mv_result.avg_delta_pct:+.2f}%)"
                )

                exp = Experiment(
                    timestamp=datetime.now(UTC).isoformat(),
                    hypothesis=experiment,
                    model=model,
                    player_type=player_type,
                    feature_diff={"added": [mv_result.candidate], "removed": []},
                    seasons={"train": train_seasons, "holdout": [holdout_season]},
                    params=params or {},
                    target_results=target_results_dict,
                    conclusion=conclusion,
                    tags=parsed_tags,
                    parent_id=parent_id,
                )

                exp_id = repo.save(exp)
                console.print(f"Logged experiment [bold green]#{exp_id}[/bold green] ({mv_result.candidate})")

            conn.commit()
    finally:
        conn.close()
