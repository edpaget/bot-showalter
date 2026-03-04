from pathlib import Path
from typing import Annotated

import typer

from fantasy_baseball_manager.cli._dispatcher import dispatch
from fantasy_baseball_manager.cli._helpers import parse_params, parse_tags
from fantasy_baseball_manager.cli._output import (
    print_ablation_result,
    print_error,
    print_gate_result,
    print_predict_result,
    print_prepare_result,
    print_system_metrics,
    print_train_result,
    print_tune_result,
)
from fantasy_baseball_manager.cli.factory import build_eval_context, build_model_context
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain import (
    Err,
    Ok,
    Projection,
    StatDistribution,
    SystemMetrics,
)
from fantasy_baseball_manager.models import (
    AblationResult,
    PredictResult,
    PrepareResult,
    TrainResult,
    TuneResult,
)
from fantasy_baseball_manager.services import GateConfig, RegressionGateRunner

_ModelArg = Annotated[str, typer.Argument(help="Name of the projection model")]
_OutputDirOpt = Annotated[str | None, typer.Option("--output-dir", help="Output directory for artifacts")]
_SeasonOpt = Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")]
_ParamOpt = Annotated[list[str] | None, typer.Option("--param", help="Model param as key=value (repeatable)")]
_VersionOpt = Annotated[str | None, typer.Option("--version", help="Run version for tracking")]
_TagOpt = Annotated[list[str] | None, typer.Option("--tag", help="Tag as key=value (repeatable)")]
_TopOpt = Annotated[int | None, typer.Option("--top", help="Top N players by WAR to include")]
_HoldoutOpt = Annotated[list[int], typer.Option("--holdout", help="Holdout season(s) to test (repeatable)")]
_BaselineOpt = Annotated[str, typer.Option("--baseline", help="Baseline version to compare against")]
_KeepOpt = Annotated[bool, typer.Option("--keep", help="Retain candidate predictions in DB after gate finishes")]


def _run_action(
    operation: str, model_name: str, output_dir: str | None, seasons: list[int] | None
) -> None:  # pragma: no cover
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


def prepare(  # pragma: no cover
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


def train(  # pragma: no cover
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


def evaluate(  # pragma: no cover
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


def predict(  # pragma: no cover
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


def finetune(model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None) -> None:  # pragma: no cover
    """Fine-tune a projection model."""
    _run_action("finetune", model, output_dir, season)


def ablate(  # pragma: no cover
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


def tune(  # pragma: no cover
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


def sweep(  # pragma: no cover
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


def gate(  # pragma: no cover
    model: _ModelArg,
    holdout: _HoldoutOpt,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    baseline: _BaselineOpt = "latest",
    top: _TopOpt = None,
    param: _ParamOpt = None,
    keep: _KeepOpt = False,
) -> None:
    """Run multi-season regression gate to verify model changes."""
    params = parse_params(param)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, model_params=params)

    with build_model_context(model, config) as model_ctx, build_eval_context(config.data_dir) as eval_ctx:
        # Validate holdouts don't overlap with training seasons
        for h in holdout:
            if h in config.seasons:
                print_error(f"Holdout season {h} overlaps with training seasons {config.seasons}")
                raise typer.Exit(code=1)

        # Validate baseline predictions exist
        baseline_projs = eval_ctx.projection_repo.get_by_system_version(model, baseline)
        if not baseline_projs:
            print_error(f"No baseline predictions found for {model}/{baseline}")
            raise typer.Exit(code=1)

        gate_config = GateConfig(
            model_name=model,
            base_training_seasons=config.seasons,
            holdout_seasons=holdout,
            baseline_system=model,
            baseline_version=baseline,
            top=top,
            model_params=params or {},
            data_dir=config.data_dir,
            artifacts_dir=config.artifacts_dir,
        )

        runner = RegressionGateRunner(
            model=model_ctx.model,
            evaluator=eval_ctx.evaluator,
            projection_repo=eval_ctx.projection_repo,
        )

        result = runner.run(gate_config)
        model_ctx.conn.commit()
        print_gate_result(result)

        if not keep:
            runner.cleanup(gate_config)
            model_ctx.conn.commit()

        if not result.passed:
            raise typer.Exit(code=1)
