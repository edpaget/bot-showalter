import json
from datetime import UTC, datetime
from statistics import mean
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._output import (
    console,
    print_error,
    print_experiment_detail,
    print_experiment_search_results,
    print_experiment_summary,
)
from fantasy_baseball_manager.cli.factory import build_experiment_context
from fantasy_baseball_manager.domain import Experiment, TargetResult
from fantasy_baseball_manager.services import summarize_exploration

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


@experiment_app.command("search")
def experiment_search(
    target: Annotated[str | None, typer.Option("--target", help="Filter by target stat")] = None,
    tag: Annotated[str | None, typer.Option("--tag", help="Filter by tag")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Filter by model")] = None,
    feature: Annotated[str | None, typer.Option("--feature", help="Filter by feature column")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Search experiments by target, tag, model, or feature."""
    with build_experiment_context(data_dir) as ctx:
        result_sets: list[set[int]] = []

        if target is not None:
            target_exps = ctx.repo.find_by_target(target)
            result_sets.append({e.id for e in target_exps})  # type: ignore[misc]

        if feature is not None:
            feature_exps = ctx.repo.find_by_feature(feature)
            result_sets.append({e.id for e in feature_exps})  # type: ignore[misc]

        if tag is not None:
            tag_exps = ctx.repo.find_by_tag(tag)
            result_sets.append({e.id for e in tag_exps})  # type: ignore[misc]

        if model is not None:
            model_exps = ctx.repo.find_by_model(model)
            result_sets.append({e.id for e in model_exps})  # type: ignore[misc]

        if result_sets:
            matching_ids = result_sets[0]
            for s in result_sets[1:]:
                matching_ids &= s
            all_exps = ctx.repo.list()
            experiments = [e for e in all_exps if e.id in matching_ids]
        else:
            experiments = ctx.repo.list()

        # Sort: by target delta_pct if filtering by target, else by avg delta_pct
        if target is not None:
            experiments.sort(key=lambda e: e.target_results.get(target, TargetResult(0, 0, 0, 0)).delta_pct)
        else:
            experiments.sort(
                key=lambda e: mean(tr.delta_pct for tr in e.target_results.values()) if e.target_results else 0.0
            )

    print_experiment_search_results(experiments, target)


@experiment_app.command("summary")
def experiment_summary_cmd(
    model: Annotated[str, typer.Option("--model", help="Model system name")],
    player_type: Annotated[str, typer.Option("--player-type", help="Player type (batter/pitcher)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show exploration summary for a model and player type."""
    with build_experiment_context(data_dir) as ctx:
        summary = summarize_exploration(ctx.repo, model, player_type)

    print_experiment_summary(summary)


@experiment_app.command("show")
def experiment_show(
    experiment_id: Annotated[int, typer.Argument(help="Experiment ID to show")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show full details of a single experiment."""
    with build_experiment_context(data_dir) as ctx:
        experiment = ctx.repo.get(experiment_id)

    if experiment is None:
        print_error(f"experiment #{experiment_id} not found")
        raise typer.Exit(code=1)

    print_experiment_detail(experiment)
