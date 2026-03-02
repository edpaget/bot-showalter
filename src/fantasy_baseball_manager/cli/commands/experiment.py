import json
from datetime import UTC, datetime
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import console, print_error
from fantasy_baseball_manager.cli.factory import build_experiment_context
from fantasy_baseball_manager.domain import Experiment, TargetResult

experiment_app = typer.Typer(name="experiment", help="Experiment journal — log and query trials")

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]


def _parse_feature_diff(raw: str) -> dict[str, list[str]]:
    """Parse '+col1,-col2,+col3' into {'added': ['col1', 'col3'], 'removed': ['col2']}."""
    added: list[str] = []
    removed: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("+"):
            added.append(token[1:])
        elif token.startswith("-"):
            removed.append(token[1:])
        else:
            added.append(token)
    return {"added": added, "removed": removed}


def _parse_seasons(raw: str) -> dict[str, list[int]]:
    """Parse '2021,2022,2023/2024' into {'train': [2021, 2022, 2023], 'holdout': [2024]}."""
    parts = raw.split("/")
    train_str = parts[0]
    holdout_str = parts[1] if len(parts) > 1 else ""
    train = [int(s.strip()) for s in train_str.split(",") if s.strip()]
    holdout = [int(s.strip()) for s in holdout_str.split(",") if s.strip()] if holdout_str else []
    return {"train": train, "holdout": holdout}


def _parse_target_results(raw: str) -> dict[str, TargetResult]:
    """Parse JSON string into dict of target -> TargetResult."""
    data = json.loads(raw)
    return {
        target: TargetResult(
            rmse=tr["rmse"],
            baseline_rmse=tr["baseline_rmse"],
            delta=tr["delta"],
            delta_pct=tr["delta_pct"],
        )
        for target, tr in data.items()
    }


@experiment_app.command("log")
def experiment_log(
    hypothesis: Annotated[str, typer.Option("--hypothesis", help="What you expected to happen")],
    model: Annotated[str, typer.Option("--model", help="Model system name")],
    player_type: Annotated[str, typer.Option("--player-type", help="Player type (batter/pitcher)")],
    conclusion: Annotated[str, typer.Option("--conclusion", help="What actually happened")],
    feature_diff: Annotated[str, typer.Option("--feature-diff", help="Feature changes: +col1,-col2")],
    seasons: Annotated[str, typer.Option("--seasons", help="Train/holdout seasons: 2021,2022/2024")],
    params: Annotated[str, typer.Option("--params", help="Hyperparameters as JSON")],
    target_results: Annotated[str, typer.Option("--target-results", help="Results as JSON")],
    tags: Annotated[str, typer.Option("--tags", help="Comma-separated tags")] = "",
    parent_id: Annotated[int | None, typer.Option("--parent-id", help="Parent experiment ID")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Log an experiment to the journal."""
    try:
        parsed_params = json.loads(params)
    except json.JSONDecodeError as e:
        print_error(f"invalid --params JSON: {e}")
        raise typer.Exit(code=1) from None

    try:
        parsed_target_results = _parse_target_results(target_results)
    except (json.JSONDecodeError, KeyError) as e:
        print_error(f"invalid --target-results JSON: {e}")
        raise typer.Exit(code=1) from None

    parsed_feature_diff = _parse_feature_diff(feature_diff)
    parsed_seasons = _parse_seasons(seasons)
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    experiment = Experiment(
        timestamp=datetime.now(UTC).isoformat(),
        hypothesis=hypothesis,
        model=model,
        player_type=player_type,
        feature_diff=parsed_feature_diff,
        seasons=parsed_seasons,
        params=parsed_params,
        target_results=parsed_target_results,
        conclusion=conclusion,
        tags=parsed_tags,
        parent_id=parent_id,
    )

    with build_experiment_context(data_dir) as ctx:
        exp_id = ctx.repo.save(experiment)
        ctx.conn.commit()

    console.print(f"Logged experiment [bold green]#{exp_id}[/bold green]")
