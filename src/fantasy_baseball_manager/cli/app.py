import math
from pathlib import Path
from typing import Annotated, Any

import typer

import fantasy_baseball_manager.models  # noqa: F401 — trigger model registration
from fantasy_baseball_manager.cli._dispatcher import dispatch
from fantasy_baseball_manager.domain.result import Err, Ok
from fantasy_baseball_manager.cli._logging import configure_logging
from fantasy_baseball_manager.cli._output import (
    console,
    print_ablation_result,
    print_adp_accuracy_report,
    print_comparison_result,
    print_dataset_list,
    print_error,
    print_features,
    print_import_result,
    print_ingest_result,
    print_player_projections,
    print_player_valuations,
    print_predict_result,
    print_prepare_result,
    print_run_detail,
    print_run_list,
    print_stratified_comparison_result,
    print_system_metrics,
    print_performance_report,
    print_system_summaries,
    print_talent_delta_report,
    print_residual_persistence_report,
    print_talent_quality_report,
    print_train_result,
    print_tune_result,
    print_value_over_adp,
    print_valuation_eval_result,
    print_valuation_rankings,
)
from fantasy_baseball_manager.cli.factory import (
    IngestContainer,
    build_adp_accuracy_context,
    build_adp_report_context,
    build_compute_container,
    build_datasets_context,
    build_eval_context,
    build_import_context,
    build_ingest_container,
    build_model_context,
    build_projections_context,
    build_report_context,
    build_runs_context,
    build_valuation_eval_context,
    build_valuations_context,
    create_model,
)
from fantasy_baseball_manager.cli.factory import EvalContext
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.config_league import load_league
from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.domain.projection_accuracy import BATTING_RATE_STATS, PITCHING_RATE_STATS
from fantasy_baseball_manager.domain.pt_normalization import build_consensus_lookup
from fantasy_baseball_manager.services.cohort import (
    assign_age_cohorts,
    assign_experience_cohorts,
    assign_top300_cohorts,
)
from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.domain.projection import Projection, StatDistribution
from fantasy_baseball_manager.ingest.adp_mapper import ingest_fantasypros_adp
from fantasy_baseball_manager.ingest.column_maps import (
    chadwick_row_to_player,
    lahman_team_row_to_team,
    make_fg_batting_mapper,
    make_fg_pitching_mapper,
    make_fg_projection_batting_mapper,
    make_fg_projection_pitching_mapper,
    make_il_stint_mapper,
    make_lahman_bio_mapper,
    make_milb_batting_mapper,
    make_position_appearance_mapper,
    make_roster_stint_mapper,
    make_sprint_speed_mapper,
    statcast_pitch_mapper,
)
from fantasy_baseball_manager.ingest.csv_source import CsvSource
from fantasy_baseball_manager.ingest.loader import Loader
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    FeatureIntrospectable,
    PredictResult,
    PrepareResult,
    TrainResult,
    TuneResult,
)
from fantasy_baseball_manager.models.registry import list_models
from fantasy_baseball_manager.models.run_manager import RunManager

app = typer.Typer(name="fbm", help="Fantasy Baseball Manager — projection model CLI")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging")] = False,
) -> None:
    """Fantasy Baseball Manager — projection model CLI."""
    configure_logging(verbose=verbose)
    if ctx.invoked_subcommand is None:
        raise typer.Exit()


_ModelArg = Annotated[str, typer.Argument(help="Name of the projection model")]
_OutputDirOpt = Annotated[str | None, typer.Option("--output-dir", help="Output directory for artifacts")]
_SeasonOpt = Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")]


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


@app.command()
def prepare(
    model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None, param: _ParamOpt = None
) -> None:
    """Prepare data for a projection model."""
    params = _parse_params(param)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, model_params=params)
    with build_model_context(model, config) as ctx:
        match dispatch("prepare", ctx.model, config):
            case Ok(PrepareResult() as r):
                print_prepare_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


_VersionOpt = Annotated[str | None, typer.Option("--version", help="Run version for tracking")]
_TagOpt = Annotated[list[str] | None, typer.Option("--tag", help="Tag as key=value (repeatable)")]


def _parse_tags(raw_tags: list[str] | None) -> dict[str, str] | None:
    if not raw_tags:
        return None
    parsed: dict[str, str] = {}
    for tag in raw_tags:
        key, _, value = tag.partition("=")
        parsed[key] = value
    return parsed


_ParamOpt = Annotated[list[str] | None, typer.Option("--param", help="Model param as key=value (repeatable)")]


def _coerce_value(value: str) -> Any:
    """Coerce a CLI string value to bool, int, float, or leave as str."""
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


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key like 'pitcher.learning_rate'."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


def _parse_params(raw_params: list[str] | None) -> dict[str, Any] | None:
    if not raw_params:
        return None
    parsed: dict[str, Any] = {}
    for param in raw_params:
        key, _, value = param.partition("=")
        _set_nested(parsed, key, _coerce_value(value))
    return parsed


@app.command()
def train(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    version: _VersionOpt = None,
    tag: _TagOpt = None,
    param: _ParamOpt = None,
) -> None:
    """Train a projection model."""
    tags = _parse_tags(tag)
    params = _parse_params(param)
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


_TopOpt = Annotated[int | None, typer.Option("--top", help="Top N players by WAR to include")]


@app.command()
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


@app.command()
def predict(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    version: _VersionOpt = None,
    tag: _TagOpt = None,
    param: _ParamOpt = None,
) -> None:
    """Generate predictions from a projection model."""
    tags = _parse_tags(tag)
    params = _parse_params(param)
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


@app.command()
def finetune(model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None) -> None:
    """Fine-tune a projection model."""
    _run_action("finetune", model, output_dir, season)


@app.command()
def ablate(
    model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None, param: _ParamOpt = None
) -> None:
    """Run ablation study on a projection model."""
    params = _parse_params(param)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, model_params=params)
    with build_model_context(model, config) as ctx:
        match dispatch("ablate", ctx.model, config):
            case Ok(AblationResult() as r):
                print_ablation_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


@app.command()
def tune(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    param: _ParamOpt = None,
    top: _TopOpt = None,
) -> None:
    """Tune hyperparameters for a projection model."""
    params = _parse_params(param)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, model_params=params, top=top)
    with build_model_context(model, config) as ctx:
        match dispatch("tune", ctx.model, config):
            case Ok(TuneResult() as r):
                print_tune_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


@app.command()
def sweep(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    param: _ParamOpt = None,
    top: _TopOpt = None,
) -> None:
    """Sweep meta-parameters (e.g. weight transforms) for a projection model."""
    params = _parse_params(param)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, model_params=params, top=top)
    with build_model_context(model, config) as ctx:
        match dispatch("sweep", ctx.model, config):
            case Ok(TuneResult() as r):
                print_tune_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


@app.command("list")
def list_cmd() -> None:
    """List all registered projection models."""
    names = list_models()
    if not names:
        console.print("No models registered.")
        return
    console.print("[bold]Registered models:[/bold]")
    for name in names:
        console.print(f"  {name}")


@app.command()
def info(model: _ModelArg) -> None:
    """Show metadata and supported operations for a model."""
    match create_model(model):
        case Ok(m):
            console.print(f"[bold]Model:[/bold] {m.name}")
            console.print(f"[bold]Description:[/bold] {m.description}")
            console.print(f"[bold]Operations:[/bold] {', '.join(sorted(m.supported_operations))}")
        case Err(e):
            print_error(e.message)
            raise typer.Exit(code=1)


@app.command()
def features(model: _ModelArg) -> None:
    """List declared features for a model."""
    match create_model(model):
        case Ok(m):
            if not isinstance(m, FeatureIntrospectable):
                print_error(f"model '{model}' does not expose features")
                raise typer.Exit(code=1)
            print_features(m.name, m.declared_features)
        case Err(e):
            print_error(e.message)
            raise typer.Exit(code=1)


@app.command("import")
def import_cmd(
    system: Annotated[str, typer.Argument(help="Projection system name (e.g. steamer, zips, atc)")],
    csv_path: Annotated[Path, typer.Argument(help="Path to CSV file")],
    version: Annotated[str, typer.Option("--version", help="Projection version")],
    player_type: Annotated[str, typer.Option("--player-type", help="Player type: batter or pitcher")],
    season: Annotated[int, typer.Option("--season", help="Season year for the projections")],
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Import third-party projections from a CSV file."""
    if not csv_path.exists():
        print_error(f"file not found: {csv_path}")
        raise typer.Exit(code=1)

    with build_import_context(data_dir) as ctx:
        players = ctx.player_repo.all()

        if player_type == "pitcher":
            mapper = make_fg_projection_pitching_mapper(
                players,
                season=season,
                system=system,
                version=version,
                source_type="third_party",
            )
        else:
            mapper = make_fg_projection_batting_mapper(
                players,
                season=season,
                system=system,
                version=version,
                source_type="third_party",
            )

        source = CsvSource(csv_path)

        def _post_upsert(projection_id: int, projection: Projection) -> None:
            if projection.distributions is not None:
                ctx.proj_repo.upsert_distributions(projection_id, list(projection.distributions.values()))

        loader = Loader(
            source, ctx.proj_repo, ctx.log_repo, mapper, "projection", conn=ctx.conn, post_upsert=_post_upsert
        )
        match loader.load(encoding="utf-8-sig"):
            case Ok(log):
                print_import_result(log)
            case Err(e):
                print_error(e.message)


def _build_cohort_assignments(ctx: EvalContext, dimension: str, season: int) -> dict[int, str]:
    """Build cohort assignment dict for the given stratification dimension."""
    if dimension == "age":
        players = ctx.player_repo.all()
        players_by_id = {p.id: p for p in players if p.id is not None}
        return assign_age_cohorts(players_by_id, season)
    if dimension == "experience":
        batting_actuals = ctx.batting_repo.get_by_season(season, source="fangraphs")
        player_ids = {a.player_id for a in batting_actuals}
        prior_batting = []
        for yr in range(season - 20, season):
            prior_batting.extend(ctx.batting_repo.get_by_season(yr, source="fangraphs"))
        return assign_experience_cohorts(prior_batting, player_ids)
    if dimension == "top300":
        batting_actuals = ctx.batting_repo.get_by_season(season, source="fangraphs")
        return assign_top300_cohorts(batting_actuals)
    msg = f"unknown dimension: {dimension}"
    raise ValueError(msg)


_STRATIFY_CHOICES = ["age", "experience", "top300"]


@app.command("compare")
def compare_cmd(
    systems: Annotated[list[str], typer.Argument(help="Systems to compare (format: system/version)")],
    season: Annotated[int, typer.Option("--season", help="Season to compare against")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to compare")] = None,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
    top: _TopOpt = None,
    stratify: Annotated[str | None, typer.Option("--stratify", help="Stratify by: age, experience, top300")] = None,
    normalize_pt: Annotated[str | None, typer.Option("--normalize-pt", help="PT source: consensus")] = None,
    rate_only: Annotated[bool, typer.Option("--rate-only", help="Evaluate rate stats only")] = False,
    min_pa: Annotated[int | None, typer.Option("--min-pa", help="Minimum PA for batters")] = None,
    min_ip: Annotated[int | None, typer.Option("--min-ip", help="Minimum IP for pitchers")] = None,
) -> None:
    """Compare multiple projection systems against actuals."""
    if stratify is not None and stratify not in _STRATIFY_CHOICES:
        print_error(f"invalid stratify dimension '{stratify}', expected one of: {', '.join(_STRATIFY_CHOICES)}")
        raise typer.Exit(code=1)

    if normalize_pt is not None and normalize_pt != "consensus":
        print_error(f"invalid --normalize-pt value '{normalize_pt}', expected 'consensus'")
        raise typer.Exit(code=1)

    parsed: list[tuple[str, str]] = []
    for s in systems:
        parts = s.split("/", 1)
        if len(parts) != 2:
            print_error(f"invalid system format '{s}', expected 'system/version'")
            raise typer.Exit(code=1)
        parsed.append((parts[0], parts[1]))

    if rate_only and stat is None:
        stat = list(BATTING_RATE_STATS + PITCHING_RATE_STATS)

    with build_eval_context(data_dir) as ctx:
        consensus = None
        if normalize_pt == "consensus":
            steamer_projs = ctx.projection_repo.get_by_season(season, system="steamer")
            zips_projs = ctx.projection_repo.get_by_season(season, system="zips")
            consensus = build_consensus_lookup(steamer_projs, zips_projs)

        if stratify is None:
            result = ctx.evaluator.compare(
                parsed, season, stats=stat, top=top, normalize_pt=consensus, min_pa=min_pa, min_ip=min_ip
            )
            print_comparison_result(result)
        else:
            cohort_assignments = _build_cohort_assignments(ctx, stratify, season)
            strat_result = ctx.evaluator.compare_stratified(
                parsed,
                season,
                cohort_assignments,
                dimension=stratify,
                stats=stat,
                top=top,
                normalize_pt=consensus,
                min_pa=min_pa,
                min_ip=min_ip,
            )
            print_stratified_comparison_result(strat_result)


# --- runs subcommand group ---

_DataDirOpt = Annotated[str, typer.Option("--data-dir", help="Data directory")]

runs_app = typer.Typer(name="runs", help="Manage first-party model runs")
app.add_typer(runs_app, name="runs")


@runs_app.command("list")
def runs_list(
    model: Annotated[str | None, typer.Option("--model", help="Filter by model name")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """List recorded model runs."""
    with build_runs_context(data_dir) as ctx:
        records = ctx.repo.list(system=model)
    print_run_list(records)


_OperationOpt = Annotated[str, typer.Option("--operation", help="Operation type (train, predict)")]


@runs_app.command("show")
def runs_show(
    run: Annotated[str, typer.Argument(help="Run identifier (system/version)")],
    operation: _OperationOpt = "train",
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show details of a model run."""
    parts = run.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid run format '{run}', expected 'system/version'")
        raise typer.Exit(code=1)
    system, version = parts

    with build_runs_context(data_dir) as ctx:
        record = ctx.repo.get(system, version, operation)
    if record is None:
        print_error(f"run '{run}' not found")
        raise typer.Exit(code=1)
    print_run_detail(record)


@runs_app.command("delete")
def runs_delete(
    run: Annotated[str, typer.Argument(help="Run identifier (system/version)")],
    operation: _OperationOpt = "train",
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Delete a model run and its artifacts."""
    parts = run.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid run format '{run}', expected 'system/version'")
        raise typer.Exit(code=1)
    system, version = parts

    with build_runs_context(data_dir) as ctx:
        record = ctx.repo.get(system, version, operation)
        if record is None:
            print_error(f"run '{run}' not found")
            raise typer.Exit(code=1)

        if not yes:
            typer.confirm(f"Delete run '{run}'?", abort=True)

        mgr = RunManager(model_run_repo=ctx.repo, artifacts_root=Path("."))
        mgr.delete_run(system, version, operation)
        ctx.conn.commit()
        console.print(f"[bold green]Deleted[/bold green] run '{run}'")


# --- datasets subcommand group ---

datasets_app = typer.Typer(name="datasets", help="Manage cached feature-set datasets")
app.add_typer(datasets_app, name="datasets")


@datasets_app.command("list")
def datasets_list(
    name: Annotated[str | None, typer.Option("--name", help="Filter by feature set name")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show all materialized feature sets and their datasets."""
    with build_datasets_context(data_dir) as ctx:
        if name:
            datasets = ctx.catalog.list_by_feature_set_name(name)
        else:
            datasets = ctx.catalog.list_all()
    print_dataset_list(datasets)


@datasets_app.command("drop")
def datasets_drop(
    name: Annotated[str | None, typer.Option("--name", help="Feature set name to drop")] = None,
    all_: Annotated[bool, typer.Option("--all", help="Drop all cached datasets")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Drop cached datasets."""
    if not name and not all_:
        print_error("provide --name or --all")
        raise typer.Exit(code=1)

    with build_datasets_context(data_dir) as ctx:
        if not yes:
            target = f"feature set '{name}'" if name else "ALL cached datasets"
            typer.confirm(f"Drop {target}?", abort=True)

        if all_:
            count = ctx.catalog.drop_all()
        else:
            assert name is not None
            count = ctx.catalog.drop_by_feature_set_name(name)

        ctx.conn.commit()

    if count == 0:
        console.print("No datasets found to drop.")
    else:
        console.print(f"[bold green]Dropped[/bold green] {count} dataset(s)")


@datasets_app.command("rebuild")
def datasets_rebuild(
    model: _ModelArg,
    season: _SeasonOpt = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Drop a model's cached datasets, then re-materialize via prepare."""
    prefix = model.replace("-", "_") + "_"

    with build_datasets_context(data_dir) as ctx:
        datasets = ctx.catalog.list_all()
        matching = [d for d in datasets if d.feature_set_name.startswith(prefix)]

        if not matching:
            console.print(f"No cached datasets found for model '{model}'.")
        else:
            if not yes:
                typer.confirm(f"Drop {len(matching)} dataset(s) for model '{model}'?", abort=True)
            count = ctx.catalog.drop_by_name_prefix(prefix)
            ctx.conn.commit()
            console.print(f"[bold green]Dropped[/bold green] {count} dataset(s)")

    config = load_config(model_name=model, seasons=season)
    with build_model_context(model, config) as ctx:
        match dispatch("prepare", ctx.model, config):
            case Ok(PrepareResult() as r):
                print_prepare_result(r)
            case Err(e):
                print_error(e.message)
                raise typer.Exit(code=1)


# --- projections subcommand group ---

projections_app = typer.Typer(name="projections", help="Look up and explore projection systems")
app.add_typer(projections_app, name="projections")


@projections_app.command("lookup")
def projections_lookup(
    player_name: Annotated[str, typer.Argument(help="Player name ('Last' or 'Last, First')")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str | None, typer.Option("--system", help="Filter by system")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Look up a player's projections across systems."""
    with build_projections_context(data_dir) as ctx:
        results = ctx.lookup_service.lookup(player_name, season, system=system)
    print_player_projections(results)


@projections_app.command("systems")
def projections_systems(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """List available projection systems for a season."""
    with build_projections_context(data_dir) as ctx:
        summaries = ctx.lookup_service.list_systems(season)
    print_system_summaries(summaries)


# --- valuations subcommand group ---

valuations_app = typer.Typer(name="valuations", help="Look up and explore player valuations")
app.add_typer(valuations_app, name="valuations")


@valuations_app.command("lookup")
def valuations_lookup(
    player_name: Annotated[str, typer.Argument(help="Player name ('Last' or 'Last, First')")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str | None, typer.Option("--system", help="Filter by valuation system")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Look up a player's valuations across systems."""
    with build_valuations_context(data_dir) as ctx:
        results = ctx.lookup_service.lookup(player_name, season, system=system)
    print_player_valuations(results)


@valuations_app.command("rankings")
def valuations_rankings(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str | None, typer.Option("--system", help="Filter by valuation system")] = None,
    player_type: Annotated[str | None, typer.Option("--player-type", help="Filter by player type")] = None,
    position: Annotated[str | None, typer.Option("--position", help="Filter by position")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N players")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show valuation rankings as a leaderboard."""
    with build_valuations_context(data_dir) as ctx:
        results = ctx.lookup_service.rankings(
            season, system=system, player_type=player_type, position=position, top=top
        )
    print_valuation_rankings(results)


@valuations_app.command("evaluate")
def valuations_evaluate(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    league_name: Annotated[str, typer.Option("--league", help="League name from fbm.toml")] = "default",
    system: Annotated[str | None, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str | None, typer.Option("--version", help="Valuation version")] = "1.0",
    top: Annotated[int | None, typer.Option("--top", help="Show top N mispricings")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate valuation accuracy against end-of-season actuals."""
    league = load_league(league_name, Path.cwd())
    with build_valuation_eval_context(data_dir) as ctx:
        result = ctx.evaluator.evaluate(system or "zar", version or "1.0", season, league)
    print_valuation_eval_result(result, top=top)


# --- ingest subcommand group ---

ingest_app = typer.Typer(name="ingest", help="Ingest historical player data and stats")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("players")
def ingest_players(
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest player data from the Chadwick register."""
    with build_ingest_container(data_dir) as container:
        source = container.player_source()
        loader = Loader(
            source,
            container.player_repo,
            container.log_repo,
            chadwick_row_to_player,
            "player",
            conn=container.conn,
        )
        match loader.load():
            case Ok(log):
                print_ingest_result(log)
            case Err(e):
                print_error(e.message)


@ingest_app.command("bio")
def ingest_bio(
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Enrich existing players with birth date, bats, and throws from Lahman."""
    with build_ingest_container(data_dir) as container:
        players = container.player_repo.all()
        source = container.bio_source()
        mapper = make_lahman_bio_mapper(players)
        loader = Loader(
            source,
            container.player_repo,
            container.log_repo,
            mapper,
            "player",
            conn=container.conn,
        )
        match loader.load():
            case Ok(log):
                print_ingest_result(log)
            case Err(e):
                print_error(e.message)


@ingest_app.command("batting")
def ingest_batting(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest historical batting stats."""
    with build_ingest_container(data_dir) as container:
        data_source = container.batting_source()
        players = container.player_repo.all()
        for yr in season:
            mapper = make_fg_batting_mapper(players)
            loader = Loader(
                data_source,
                container.batting_stats_repo,
                container.log_repo,
                mapper,
                "batting_stats",
                conn=container.conn,
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("pitching")
def ingest_pitching(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest historical pitching stats."""
    with build_ingest_container(data_dir) as container:
        data_source = container.pitching_source()
        players = container.player_repo.all()
        for yr in season:
            mapper = make_fg_pitching_mapper(players)
            loader = Loader(
                data_source,
                container.pitching_stats_repo,
                container.log_repo,
                mapper,
                "pitching_stats",
                conn=container.conn,
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("statcast")
def ingest_statcast(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest Statcast pitch-level data."""
    with build_ingest_container(data_dir) as container:
        for yr in season:
            start_dt = f"{yr}-03-01"
            end_dt = f"{yr}-11-30"
            loader = Loader(
                container.statcast_source(),
                container.statcast_pitch_repo,
                container.log_repo,
                statcast_pitch_mapper,
                "statcast_pitch",
                conn=container.statcast_conn,
                log_conn=container.conn,
            )
            match loader.load(start_dt=start_dt, end_dt=end_dt):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("sprint-speed")
def ingest_sprint_speed(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest Baseball Savant sprint speed data."""
    with build_ingest_container(data_dir) as container:
        for yr in season:
            mapper = make_sprint_speed_mapper(season=yr)
            loader = Loader(
                container.sprint_speed_source(),
                container.sprint_speed_repo,
                container.log_repo,
                mapper,
                "sprint_speed",
                conn=container.statcast_conn,
                log_conn=container.conn,
            )
            match loader.load(year=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("il")
def ingest_il(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s) to ingest (repeatable)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest IL transaction data from the MLB Stats API."""
    with build_ingest_container(data_dir) as container:
        players = container.player_repo.all()
        for yr in season:
            mapper = make_il_stint_mapper(players, season=yr)
            source = container.il_source()
            loader = Loader(
                source,
                container.il_stint_repo,
                container.log_repo,
                mapper,
                "il_stint",
                conn=container.conn,
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("appearances")
def ingest_appearances(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest position appearance data from Lahman."""
    with build_ingest_container(data_dir) as container:
        players = container.player_repo.all()
        mapper = make_position_appearance_mapper(players)
        for yr in season:
            loader = Loader(
                container.appearances_source(),
                container.position_appearance_repo,
                container.log_repo,
                mapper,
                "position_appearance",
                conn=container.conn,
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


@ingest_app.command("roster")
def ingest_roster(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest roster stint data from Lahman."""
    with build_ingest_container(data_dir) as container:
        # Auto-upsert teams first
        teams_source = container.teams_source()
        for yr in season:
            teams_rows = teams_source.fetch(season=yr)
            for row in teams_rows:
                team = lahman_team_row_to_team(row)
                if team is not None:
                    container.team_repo.upsert(team)
            container.conn.commit()

        players = container.player_repo.all()
        teams = container.team_repo.all()
        mapper = make_roster_stint_mapper(players, teams)
        for yr in season:
            loader = Loader(
                container.appearances_source(),
                container.roster_stint_repo,
                container.log_repo,
                mapper,
                "roster_stint",
                conn=container.conn,
            )
            match loader.load(season=yr):
                case Ok(log):
                    print_ingest_result(log)
                case Err(e):
                    print_error(e.message)
                    continue


def _auto_register_players(rows: list[dict[str, Any]], container: IngestContainer) -> None:
    """Register any players in the rows that aren't already in the player table."""
    if not rows:
        return
    existing_mlbam_ids = {p.mlbam_id for p in container.player_repo.all() if p.mlbam_id is not None}
    registered = 0
    for row in rows:
        mlbam_id = row.get("mlbam_id")
        if mlbam_id is None or (isinstance(mlbam_id, float) and math.isnan(mlbam_id)):
            continue
        if int(mlbam_id) in existing_mlbam_ids:
            continue
        first = row.get("first_name", "")
        last = row.get("last_name", "")
        container.player_repo.upsert(Player(name_first=str(first), name_last=str(last), mlbam_id=int(mlbam_id)))
        existing_mlbam_ids.add(int(mlbam_id))
        registered += 1
    if registered:
        container.conn.commit()


class _PreloadedSource:
    """Wraps pre-fetched rows so Loader can use them without re-fetching."""

    def __init__(self, rows: list[dict[str, Any]], original: object) -> None:
        self._rows = rows
        self._source_type = getattr(original, "source_type", "unknown")
        self._source_detail = getattr(original, "source_detail", "unknown")

    @property
    def source_type(self) -> str:
        return self._source_type

    @property
    def source_detail(self) -> str:
        return self._source_detail

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        return self._rows


_MILB_LEVELS = ["AAA", "AA", "A+", "A", "ROK"]


@ingest_app.command("milb-batting")
def ingest_milb_batting(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    level: Annotated[list[str] | None, typer.Option("--level", help="Level(s): AAA, AA, A+, A, ROK")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest minor league batting stats from the MLB Stats API."""
    levels = level if level else _MILB_LEVELS
    with build_ingest_container(data_dir) as container:
        for yr in season:
            for lvl in levels:
                source = container.milb_batting_source()
                rows = source.fetch(season=yr, level=lvl)
                _auto_register_players(rows, container)
                players = container.player_repo.all()
                mapper = make_milb_batting_mapper(players)
                loader = Loader(
                    _PreloadedSource(rows, source),
                    container.minor_league_batting_stats_repo,
                    container.log_repo,
                    mapper,
                    "minor_league_batting_stats",
                    conn=container.conn,
                )
                match loader.load(season=yr, level=lvl):
                    case Ok(log):
                        print_ingest_result(log)
                    case Err(e):
                        print_error(e.message)
                        continue


@ingest_app.command("adp")
def ingest_adp(
    csv_path: Annotated[Path, typer.Argument(help="Path to FantasyPros ADP CSV")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    as_of: Annotated[str | None, typer.Option("--as-of", help="Snapshot date (ISO)")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Ingest ADP data from a FantasyPros CSV file."""
    if not csv_path.exists():
        print_error(f"file not found: {csv_path}")
        raise typer.Exit(code=1)

    with build_ingest_container(data_dir) as container:
        source = CsvSource(csv_path)
        rows = source.fetch()
        players = container.player_repo.all()
        result = ingest_fantasypros_adp(rows, container.adp_repo, players, season=season, as_of=as_of)
        container.conn.commit()
        console.print(f"  Loaded {result.loaded} ADP records, skipped {result.skipped}")
        if result.unmatched:
            console.print(
                f"  [yellow]Unmatched players ({len(result.unmatched)}):[/yellow] {', '.join(result.unmatched[:20])}"
            )
            if len(result.unmatched) > 20:
                console.print(f"  ... and {len(result.unmatched) - 20} more")


@ingest_app.command("adp-bulk")
def ingest_adp_bulk(
    directory: Annotated[Path, typer.Argument(help="Directory with fantasypros_*.csv files")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Bulk-ingest ADP data from all fantasypros_*.csv files in a directory."""
    if not directory.is_dir():
        print_error(f"not a directory: {directory}")
        raise typer.Exit(code=1)

    csv_files = sorted(directory.glob("fantasypros_*.csv"))
    if not csv_files:
        print_error(f"no fantasypros_*.csv files found in {directory}")
        raise typer.Exit(code=1)

    with build_ingest_container(data_dir) as container:
        players = container.player_repo.all()
        for csv_file in csv_files:
            stem = csv_file.stem
            year_str = stem.replace("fantasypros_", "")
            try:
                season = int(year_str)
            except ValueError:
                console.print(f"  [yellow]Skipping {csv_file.name}: cannot parse season[/yellow]")
                continue

            source = CsvSource(csv_file)
            rows = source.fetch()
            result = ingest_fantasypros_adp(rows, container.adp_repo, players, season=season)
            container.conn.commit()
            n_unmatched = len(result.unmatched)
            console.print(
                f"  {csv_file.name}: loaded {result.loaded}, skipped {result.skipped}, unmatched {n_unmatched}"
            )


# --- compute subcommand group ---

_MILB_COMPUTE_LEVELS = ["AAA", "AA", "A+", "A", "ROK"]

compute_app = typer.Typer(name="compute", help="Compute derived data from ingested stats")
app.add_typer(compute_app, name="compute")


@compute_app.command("league-env")
def compute_league_env(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    level: Annotated[list[str] | None, typer.Option("--level", help="Level(s): AAA, AA, A+, A, ROK")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Compute league environment aggregates from minor league batting stats."""
    levels = level if level else _MILB_COMPUTE_LEVELS
    with build_compute_container(data_dir) as container:
        for yr in season:
            for lvl in levels:
                count = container.league_environment_service.compute_for_season_level(yr, lvl)
                container.conn.commit()
                console.print(f"  {lvl} {yr}: {count} league(s) computed")
    console.print("[bold green]Done.[/bold green]")


# --- report subcommand group ---

report_app = typer.Typer(name="report", help="Over/underperformance reports vs model predictions")
app.add_typer(report_app, name="report")


@report_app.command("overperformers")
def report_overperformers(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to report")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N rows")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show players who outperformed their expected stats."""
    parts = system.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid system format '{system}', expected 'system/version'")
        raise typer.Exit(code=1)
    sys_name, version = parts

    with build_report_context(data_dir) as ctx:
        deltas = ctx.report_service.compute_deltas(
            sys_name,
            version,
            season,
            player_type,
            stats=stat,
        )

    overperformers = [d for d in deltas if d.performance_delta > 0]
    overperformers.sort(key=lambda d: d.performance_delta, reverse=True)
    if top is not None:
        overperformers = overperformers[:top]
    print_performance_report("Overperformers", overperformers)


@report_app.command("underperformers")
def report_underperformers(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to report")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N rows")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show players who underperformed their expected stats."""
    parts = system.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid system format '{system}', expected 'system/version'")
        raise typer.Exit(code=1)
    sys_name, version = parts

    with build_report_context(data_dir) as ctx:
        deltas = ctx.report_service.compute_deltas(
            sys_name,
            version,
            season,
            player_type,
            stats=stat,
        )

    underperformers = [d for d in deltas if d.performance_delta < 0]
    underperformers.sort(key=lambda d: d.performance_delta)
    if top is not None:
        underperformers = underperformers[:top]
    print_performance_report("Underperformers", underperformers)


@report_app.command("talent-delta")
def report_talent_delta(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[int, typer.Option("--season", help="Season year")],
    player_type: Annotated[str, typer.Option("--player-type", help="batter or pitcher")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to include")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N per direction per stat")] = None,
    min_pa: Annotated[int | None, typer.Option("--min-pa", help="Minimum PA (batters) or IP (pitchers)")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show talent-delta report: regression candidates and buy-low targets."""
    parts = system.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid system format '{system}', expected 'system/version'")
        raise typer.Exit(code=1)
    sys_name, version = parts

    with build_report_context(data_dir) as ctx:
        deltas = ctx.report_service.compute_deltas(
            sys_name,
            version,
            season,
            player_type,
            stats=stat,
            min_pa=min_pa,
        )

    pa_label = "IP" if player_type == "pitcher" else "PA"
    min_pa_str = f", min {min_pa} {pa_label}" if min_pa else ""
    title = f"Talent Delta — {system} vs {season} actuals ({player_type}s{min_pa_str})"
    print_talent_delta_report(title, deltas, top=top)


@report_app.command("talent-quality")
def report_talent_quality(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[list[int], typer.Option("--season", help="Two seasons (N and N+1)")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to evaluate")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate true-talent estimation quality across consecutive seasons."""
    if len(season) != 2:
        print_error("exactly two --season values required (N and N+1)")
        raise typer.Exit(code=1)

    parts = system.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid system format '{system}', expected 'system/version'")
        raise typer.Exit(code=1)
    sys_name, version = parts

    season_n, season_n1 = sorted(season)

    with build_report_context(data_dir) as ctx:
        reports = []
        for player_type in ("batter", "pitcher"):
            report = ctx.talent_evaluator.evaluate(sys_name, version, season_n, season_n1, player_type, stats=stat)
            if report.stat_metrics:
                reports.append(report)

    if not reports:
        console.print("No projections found for either player type.")
        raise typer.Exit(code=1)

    print_talent_quality_report(reports)


@report_app.command("residual-persistence")
def report_residual_persistence(
    system: Annotated[str, typer.Argument(help="System/version (e.g. statcast-gbm/latest)")],
    season: Annotated[list[int], typer.Option("--season", help="Two seasons (N and N+1)")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to evaluate")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Diagnose residual persistence for batter projections across consecutive seasons."""
    if len(season) != 2:
        print_error("exactly two --season values required (N and N+1)")
        raise typer.Exit(code=1)

    parts = system.split("/", 1)
    if len(parts) != 2:
        print_error(f"invalid system format '{system}', expected 'system/version'")
        raise typer.Exit(code=1)
    sys_name, version = parts

    season_n, season_n1 = sorted(season)

    with build_report_context(data_dir) as ctx:
        report = ctx.residual_diagnostic.diagnose(sys_name, version, season_n, season_n1, stats=stat)

    if not report.stat_metrics:
        console.print("No batter projections found.")
        raise typer.Exit(code=1)

    print_residual_persistence_report(report)


@report_app.command("value-over-adp")
def report_value_over_adp(
    season: Annotated[int, typer.Option("--season", help="Season year")],
    system: Annotated[str, typer.Option("--system", help="Valuation system")] = "zar",
    version: Annotated[str, typer.Option("--version", help="Valuation version")] = "1.0",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    player_type: Annotated[str | None, typer.Option("--player-type")] = None,
    position: Annotated[str | None, typer.Option("--position")] = None,
    top: Annotated[int | None, typer.Option("--top", help="Show top N per section")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show value-over-ADP report: buy targets, avoids, and sleepers."""
    with build_adp_report_context(data_dir) as ctx:
        report = ctx.service.compute_value_over_adp(
            season,
            system,
            version,
            provider=provider,
            player_type=player_type,
            position=position,
            top=top,
        )
    print_value_over_adp(report)


@report_app.command("adp-accuracy")
def report_adp_accuracy(
    season: Annotated[list[int], typer.Option("--season", help="Season year(s)")],
    league_name: Annotated[str, typer.Option("--league", help="League name")] = "default",
    provider: Annotated[str, typer.Option("--provider", help="ADP provider")] = "fantasypros",
    compare_system: Annotated[str | None, typer.Option("--compare-system", help="Valuation system/version")] = None,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Evaluate how well ADP predicts actual season outcomes."""
    league = load_league(league_name, Path.cwd())

    compare: tuple[str, str] | None = None
    if compare_system is not None:
        parts = compare_system.split("/", 1)
        if len(parts) != 2:
            print_error(f"invalid compare-system format '{compare_system}', expected 'system/version'")
            raise typer.Exit(code=1)
        compare = (parts[0], parts[1])

    with build_adp_accuracy_context(data_dir) as ctx:
        report = ctx.evaluator.evaluate(season, league, provider=provider, compare_system=compare)
    print_adp_accuracy_report(report)
