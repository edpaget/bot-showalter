from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._dispatcher import dispatch
from fantasy_baseball_manager.cli._helpers import parse_params, parse_tags
from fantasy_baseball_manager.cli._output import (
    print_ablation_result,
    print_error,
    print_predict_result,
    print_prepare_result,
    print_system_metrics,
    print_train_result,
    print_tune_result,
)
from fantasy_baseball_manager.cli.factory import build_model_context
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.domain.projection import Projection, StatDistribution
from fantasy_baseball_manager.domain.result import Err, Ok
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    PredictResult,
    PrepareResult,
    TrainResult,
    TuneResult,
)

_ModelArg = Annotated[str, typer.Argument(help="Name of the projection model")]
_OutputDirOpt = Annotated[str | None, typer.Option("--output-dir", help="Output directory for artifacts")]
_SeasonOpt = Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")]
_ParamOpt = Annotated[list[str] | None, typer.Option("--param", help="Model param as key=value (repeatable)")]
_VersionOpt = Annotated[str | None, typer.Option("--version", help="Run version for tracking")]
_TagOpt = Annotated[list[str] | None, typer.Option("--tag", help="Tag as key=value (repeatable)")]
_TopOpt = Annotated[int | None, typer.Option("--top", help="Top N players by WAR to include")]


def _run_action(operation: str, model_name: str, output_dir: str | None, seasons: list[int] | None) -> None:
    config = load_config(model_name=model_name, output_dir=output_dir, seasons=seasons)
    with build_model_context(model_name, config) as ctx:
        match dispatch(operation, ctx.model, config):
            case Ok(PrepareResult() as r):
                print_prepare_result(r)
            case Ok(TrainResult() as r):
                print_train_result(r)
            case Ok(SystemMetrics() as r):
                print_system_metrics(r)
            case Ok(PredictResult() as r):
                print_predict_result(r)
            case Ok(AblationResult() as r):
                print_ablation_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


def prepare(
    model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None, param: _ParamOpt = None
) -> None:
    """Prepare data for a projection model."""
    params = parse_params(param)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, model_params=params)
    with build_model_context(model, config) as ctx:
        match dispatch("prepare", ctx.model, config):
            case Ok(PrepareResult() as r):
                print_prepare_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


def train(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    version: _VersionOpt = None,
    tag: _TagOpt = None,
    param: _ParamOpt = None,
) -> None:
    """Train a projection model."""
    tags = parse_tags(tag)
    params = parse_params(param)
    config = load_config(
        model_name=model, output_dir=output_dir, seasons=season, version=version, tags=tags, model_params=params
    )

    with build_model_context(model, config) as ctx:
        match dispatch("train", ctx.model, config, run_manager=ctx.run_manager):
            case Ok(TrainResult() as r):
                if config.version is not None:
                    ctx.conn.commit()
                print_train_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


def evaluate(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    top: _TopOpt = None,
) -> None:
    """Evaluate a projection model."""
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, top=top)
    with build_model_context(model, config) as ctx:
        match dispatch("evaluate", ctx.model, config):
            case Ok(SystemMetrics() as r):
                print_system_metrics(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


def predict(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    version: _VersionOpt = None,
    tag: _TagOpt = None,
    param: _ParamOpt = None,
) -> None:
    """Generate predictions from a projection model."""
    tags = parse_tags(tag)
    params = parse_params(param)
    if params and "league" in params and isinstance(params["league"], str):
        params["league"] = load_league(params["league"], Path.cwd())
    config = load_config(
        model_name=model, output_dir=output_dir, seasons=season, version=version, tags=tags, model_params=params
    )

    with build_model_context(model, config) as ctx:
        match dispatch("predict", ctx.model, config, run_manager=ctx.run_manager):
            case Ok(PredictResult() as result):
                if ctx.projection_repo is not None:
                    version = config.version or "latest"
                    projection_ids: dict[tuple[int, str], int] = {}
                    for pred in result.predictions:
                        if "player_id" not in pred or "season" not in pred:
                            continue
                        stat_json = {k: v for k, v in pred.items() if k not in ("player_id", "season", "player_type")}
                        proj = Projection(
                            player_id=pred["player_id"],
                            season=pred["season"],
                            system=model,
                            version=version,
                            player_type=pred.get("player_type", "batter"),
                            stat_json=stat_json,
                        )
                        proj_id = ctx.projection_repo.upsert(proj)
                        projection_ids[(pred["player_id"], pred.get("player_type", "batter"))] = proj_id

                    if result.distributions is not None:
                        grouped_dists: dict[tuple[int, str], list[StatDistribution]] = {}
                        for dist_dict in result.distributions:
                            key = (dist_dict["player_id"], dist_dict["player_type"])
                            sd = StatDistribution(
                                stat=dist_dict["stat"],
                                p10=dist_dict["p10"],
                                p25=dist_dict["p25"],
                                p50=dist_dict["p50"],
                                p75=dist_dict["p75"],
                                p90=dist_dict["p90"],
                                mean=dist_dict["mean"],
                                std=dist_dict["std"],
                            )
                            grouped_dists.setdefault(key, []).append(sd)

                        for key, dists in grouped_dists.items():
                            proj_id = projection_ids.get(key)
                            if proj_id is not None:
                                ctx.projection_repo.upsert_distributions(proj_id, dists)

                ctx.conn.commit()
                print_predict_result(result)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


def finetune(model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None) -> None:
    """Fine-tune a projection model."""
    _run_action("finetune", model, output_dir, season)


def ablate(
    model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None, param: _ParamOpt = None
) -> None:
    """Run ablation study on a projection model."""
    params = parse_params(param)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, model_params=params)
    with build_model_context(model, config) as ctx:
        match dispatch("ablate", ctx.model, config):
            case Ok(AblationResult() as r):
                print_ablation_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


def tune(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    param: _ParamOpt = None,
    top: _TopOpt = None,
) -> None:
    """Tune hyperparameters for a projection model."""
    params = parse_params(param)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, model_params=params, top=top)
    with build_model_context(model, config) as ctx:
        match dispatch("tune", ctx.model, config):
            case Ok(TuneResult() as r):
                print_tune_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


def sweep(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    param: _ParamOpt = None,
    top: _TopOpt = None,
) -> None:
    """Sweep meta-parameters (e.g. weight transforms) for a projection model."""
    params = parse_params(param)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, model_params=params, top=top)
    with build_model_context(model, config) as ctx:
        match dispatch("sweep", ctx.model, config):
            case Ok(TuneResult() as r):
                print_tune_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)
