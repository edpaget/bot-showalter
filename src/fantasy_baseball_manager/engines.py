import typer

from fantasy_baseball_manager.pipeline.presets import PIPELINES

SUPPORTED_ENGINES: tuple[str, ...] = tuple(PIPELINES.keys())
DEFAULT_ENGINE: str = "marcel_statreg"

SUPPORTED_METHODS: tuple[str, ...] = ("zscore",)
DEFAULT_METHOD: str = "zscore"


def validate_engine(engine: str) -> None:
    if engine not in SUPPORTED_ENGINES:
        typer.echo(f"Unknown engine: {engine}", err=True)
        raise typer.Exit(code=1)


def validate_method(method: str) -> None:
    if method not in SUPPORTED_METHODS:
        typer.echo(f"Unknown method: {method}", err=True)
        raise typer.Exit(code=1)
