import typer

from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    EvalResult,
    PredictResult,
    PrepareResult,
    TrainResult,
)


def print_prepare_result(result: PrepareResult) -> None:
    typer.echo(f"Prepared model '{result.model_name}'")
    typer.echo(f"  Rows processed: {result.rows_processed}")
    typer.echo(f"  Artifacts: {result.artifacts_path}")


def print_train_result(result: TrainResult) -> None:
    typer.echo(f"Trained model '{result.model_name}'")
    if result.metrics:
        for name, value in result.metrics.items():
            typer.echo(f"  {name}: {value}")
    typer.echo(f"  Artifacts: {result.artifacts_path}")


def print_eval_result(result: EvalResult) -> None:
    typer.echo(f"Evaluated model '{result.model_name}'")
    if result.metrics:
        for name, value in result.metrics.items():
            typer.echo(f"  {name}: {value}")


def print_predict_result(result: PredictResult) -> None:
    typer.echo(f"Predictions from model '{result.model_name}'")
    typer.echo(f"  {len(result.predictions)} predictions written to {result.output_path}")


def print_ablation_result(result: AblationResult) -> None:
    typer.echo(f"Ablation results for model '{result.model_name}'")
    if result.feature_impacts:
        for feature, impact in sorted(result.feature_impacts.items(), key=lambda x: -abs(x[1])):
            typer.echo(f"  {feature}: {impact:+.4f}")
