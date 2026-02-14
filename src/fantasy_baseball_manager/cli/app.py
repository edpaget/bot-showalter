from pathlib import Path
from typing import Annotated

import typer

import fantasy_baseball_manager.models  # noqa: F401 — trigger model registration
from fantasy_baseball_manager.cli._dispatcher import UnsupportedOperation, dispatch
from fantasy_baseball_manager.cli._output import (
    print_ablation_result,
    print_eval_result,
    print_predict_result,
    print_prepare_result,
    print_train_result,
)
from fantasy_baseball_manager.config import load_config
from fantasy_baseball_manager.db.connection import create_connection
from fantasy_baseball_manager.features.assembler import SqliteDatasetAssembler
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    EvalResult,
    PredictResult,
    PrepareResult,
    ProjectionModel,
    TrainResult,
)
from fantasy_baseball_manager.models.registry import get, list_models

app = typer.Typer(name="fbm", help="Fantasy Baseball Manager — projection model CLI")

_ModelArg = Annotated[str, typer.Argument(help="Name of the projection model")]
_OutputDirOpt = Annotated[str | None, typer.Option("--output-dir", help="Output directory for artifacts")]
_SeasonOpt = Annotated[list[int] | None, typer.Option("--season", help="Season year(s) to include")]


_ASSEMBLER_OPERATIONS: frozenset[str] = frozenset({"prepare"})


def _run_action(operation: str, model_name: str, output_dir: str | None, seasons: list[int] | None) -> None:
    config = load_config(model_name=model_name, output_dir=output_dir, seasons=seasons)
    assembler = None
    if operation in _ASSEMBLER_OPERATIONS:
        conn = create_connection(Path(config.data_dir) / "fbm.db")
        assembler = SqliteDatasetAssembler(conn)
    try:
        result = dispatch(operation, model_name, config, assembler=assembler)
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


@app.command()
def train(model: _ModelArg, output_dir: _OutputDirOpt = None, season: _SeasonOpt = None) -> None:
    """Train a projection model."""
    _run_action("train", model, output_dir, season)


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
        m: ProjectionModel = get(model)
    except KeyError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"Model: {m.name}")
    typer.echo(f"Description: {m.description}")
    typer.echo(f"Operations: {', '.join(sorted(m.supported_operations))}")
