import typer

SUPPORTED_ENGINES: tuple[str, ...] = ("marcel",)
DEFAULT_ENGINE: str = "marcel"

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
