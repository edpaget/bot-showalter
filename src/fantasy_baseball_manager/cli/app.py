from pathlib import Path
from typing import Annotated

import typer

import fantasy_baseball_manager.models  # noqa: F401 — trigger model registration
from fantasy_baseball_manager.cli._dispatcher import UnsupportedOperation, dispatch
from fantasy_baseball_manager.cli._output import (
    print_ablation_result,
    print_comparison_result,
    print_eval_result,
    print_features,
    print_import_result,
    print_player_projections,
    print_predict_result,
    print_prepare_result,
    print_run_detail,
    print_run_list,
    print_system_metrics,
    print_system_summaries,
    print_train_result,
)
from fantasy_baseball_manager.cli.factory import (
    build_eval_context,
    build_import_context,
    build_model_context,
    build_projections_context,
    build_runs_context,
    create_model,
)
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.ingest.column_maps import (
    make_fg_projection_batting_mapper,
    make_fg_projection_pitching_mapper,
)
from fantasy_baseball_manager.ingest.csv_source import CsvSource
from fantasy_baseball_manager.ingest.loader import StatsLoader
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    EvalResult,
    FeatureIntrospectable,
    PredictResult,
    PrepareResult,
    ProjectionModel,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import list_models
from fantasy_baseball_manager.models.run_manager import RunManager

app = typer.Typer(name="fbm", help="Fantasy Baseball Manager — projection model CLI")

_ModelArg = Annotated[str, typer.Argument(help="Name of the projection model")]
_OutputDirOpt = Annotated[str | None, typer.Option("--output-dir", help="Output directory for artifacts")]
_SeasonOpt = Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")]


def _run_action(operation: str, model_name: str, output_dir: str | None, seasons: list[int] | None) -> None:
    config = load_config(model_name=model_name, output_dir=output_dir, seasons=seasons)
    with build_model_context(model_name, config) as ctx:
        try:
            result = dispatch(operation, ctx.model, config)
        except UnsupportedOperation as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1) from None

    match result:
        case PrepareResult():
            print_prepare_result(result)
        case TrainResult():
            print_train_result(result)
        case EvalResult():
            print_eval_result(result)
        case PredictResult():
            print_predict_result(result)
        case AblationResult():
            print_ablation_result(result)


@app.command()
def prepare(model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None) -> None:
    """Prepare data for a projection model."""
    _run_action("prepare", model, output_dir, season)


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


@app.command()
def train(
    model: _ModelArg,
    output_dir: _OutputDirOpt = None,
    season: _SeasonOpt = None,
    version: _VersionOpt = None,
    tag: _TagOpt = None,
) -> None:
    """Train a projection model."""
    tags = _parse_tags(tag)
    config = load_config(model_name=model, output_dir=output_dir, seasons=season, version=version, tags=tags)

    with build_model_context(model, config) as ctx:
        try:
            result = dispatch("train", ctx.model, config, run_manager=ctx.run_manager)
            if config.version is not None:
                ctx.conn.commit()
        except UnsupportedOperation as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1) from None

    match result:
        case TrainResult():
            print_train_result(result)


@app.command()
def evaluate(model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None) -> None:
    """Evaluate a projection model."""
    _run_action("evaluate", model, output_dir, season)


@app.command()
def predict(model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None) -> None:
    """Generate predictions from a projection model."""
    _run_action("predict", model, output_dir, season)


@app.command()
def finetune(model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None) -> None:
    """Fine-tune a projection model."""
    _run_action("finetune", model, output_dir, season)


@app.command()
def ablate(model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None) -> None:
    """Run ablation study on a projection model."""
    _run_action("ablate", model, output_dir, season)


@app.command("list")
def list_cmd() -> None:
    """List all registered projection models."""
    names = list_models()
    if not names:
        typer.echo("No models registered.")
        return
    typer.echo("Registered models:")
    for name in names:
        typer.echo(f"  {name}")


@app.command()
def info(model: _ModelArg) -> None:
    """Show metadata and supported operations for a model."""
    try:
        m: ProjectionModel = create_model(model)
    except KeyError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Model: {m.name}")
    typer.echo(f"Description: {m.description}")
    typer.echo(f"Operations: {', '.join(sorted(m.supported_operations))}")


@app.command()
def features(model: _ModelArg) -> None:
    """List declared features for a model."""
    try:
        m: ProjectionModel = create_model(model)
    except KeyError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None

    if not isinstance(m, FeatureIntrospectable):
        typer.echo(f"Error: model '{model}' does not expose features", err=True)
        raise typer.Exit(code=1)

    print_features(m.name, m.declared_features)


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
        typer.echo(f"Error: file not found: {csv_path}", err=True)
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
        loader = StatsLoader(source, ctx.proj_repo, ctx.log_repo, mapper, "projection", conn=ctx.conn)
        log = loader.load(encoding="utf-8-sig")
    print_import_result(log)


@app.command("eval")
def eval_cmd(
    system: Annotated[str, typer.Argument(help="Projection system name")],
    version: Annotated[str, typer.Option("--version", help="Projection version")],
    season: Annotated[int, typer.Option("--season", help="Season to evaluate against")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to evaluate")] = None,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Evaluate a projection system against actual stats."""
    with build_eval_context(data_dir) as ctx:
        result = ctx.evaluator.evaluate(system, version, season, stats=stat)
    print_system_metrics(result)


@app.command("compare")
def compare_cmd(
    systems: Annotated[list[str], typer.Argument(help="Systems to compare (format: system/version)")],
    season: Annotated[int, typer.Option("--season", help="Season to compare against")],
    stat: Annotated[list[str] | None, typer.Option("--stat", help="Stat(s) to compare")] = None,
    data_dir: Annotated[str, typer.Option("--data-dir", help="Data directory")] = "./data",
) -> None:
    """Compare multiple projection systems against actuals."""
    parsed: list[tuple[str, str]] = []
    for s in systems:
        parts = s.split("/", 1)
        if len(parts) != 2:
            typer.echo(f"Error: invalid system format '{s}', expected 'system/version'", err=True)
            raise typer.Exit(code=1)
        parsed.append((parts[0], parts[1]))

    with build_eval_context(data_dir) as ctx:
        result = ctx.evaluator.compare(parsed, season, stats=stat)
    print_comparison_result(result)


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


@runs_app.command("show")
def runs_show(
    run: Annotated[str, typer.Argument(help="Run identifier (system/version)")],
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Show details of a model run."""
    parts = run.split("/", 1)
    if len(parts) != 2:
        typer.echo(f"Error: invalid run format '{run}', expected 'system/version'", err=True)
        raise typer.Exit(code=1)
    system, version = parts

    with build_runs_context(data_dir) as ctx:
        record = ctx.repo.get(system, version)
    if record is None:
        typer.echo(f"Error: run '{run}' not found", err=True)
        raise typer.Exit(code=1)
    print_run_detail(record)


@runs_app.command("delete")
def runs_delete(
    run: Annotated[str, typer.Argument(help="Run identifier (system/version)")],
    yes: Annotated[bool, typer.Option("--yes", help="Skip confirmation")] = False,
    data_dir: _DataDirOpt = "./data",
) -> None:
    """Delete a model run and its artifacts."""
    parts = run.split("/", 1)
    if len(parts) != 2:
        typer.echo(f"Error: invalid run format '{run}', expected 'system/version'", err=True)
        raise typer.Exit(code=1)
    system, version = parts

    with build_runs_context(data_dir) as ctx:
        record = ctx.repo.get(system, version)
        if record is None:
            typer.echo(f"Error: run '{run}' not found", err=True)
            raise typer.Exit(code=1)

        if not yes:
            typer.confirm(f"Delete run '{run}'?", abort=True)

        mgr = RunManager(model_run_repo=ctx.repo, artifacts_root=Path("."))
        mgr.delete_run(system, version)
        ctx.conn.commit()
        typer.echo(f"Deleted run '{run}'")


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
