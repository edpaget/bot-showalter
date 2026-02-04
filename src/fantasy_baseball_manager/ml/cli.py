"""CLI commands for ML model training and management."""

from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()

ml_app = typer.Typer(help="Machine learning model commands.")


@ml_app.command(name="train")
def train_cmd(
    years: Annotated[
        str,
        typer.Option(
            "--years",
            "-y",
            help="Comma-separated target years for training (e.g., 2020,2021,2022,2023)",
        ),
    ],
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="Name for the trained model",
        ),
    ] = "default",
    pipeline: Annotated[
        str,
        typer.Option(
            "--pipeline",
            "-p",
            help="Base pipeline to use for generating projections (marcel or marcel_full)",
        ),
    ] = "marcel",
) -> None:
    """Train gradient boosting residual models on historical data.

    Example:
        uv run python -m fantasy_baseball_manager ml train --years 2020,2021,2022,2023 --name default
    """
    from fantasy_baseball_manager.cache.factory import create_cache_store
    from fantasy_baseball_manager.marcel.data_source import CachedStatsDataSource, PybaseballDataSource
    from fantasy_baseball_manager.ml.persistence import ModelStore
    from fantasy_baseball_manager.ml.training import ResidualModelTrainer
    from fantasy_baseball_manager.pipeline.batted_ball_data import (
        CachedBattedBallDataSource,
        PybaseballBattedBallDataSource,
    )
    from fantasy_baseball_manager.pipeline.presets import build_pipeline
    from fantasy_baseball_manager.pipeline.skill_data import (
        CachedSkillDataSource,
        CompositeSkillDataSource,
        FanGraphsSkillDataSource,
        StatcastSprintSpeedSource,
    )
    from fantasy_baseball_manager.pipeline.statcast_data import (
        CachedStatcastDataSource,
        PybaseballStatcastDataSource,
    )
    from fantasy_baseball_manager.player_id.mapper import build_cached_sfbb_mapper

    # Parse years
    target_years = tuple(int(y.strip()) for y in years.split(","))
    typer.echo(f"Training models for target years: {target_years}")

    # Build pipeline
    typer.echo(f"Using pipeline: {pipeline}")
    proj_pipeline = build_pipeline(pipeline)

    # Setup data sources
    cache = create_cache_store()
    data_source = CachedStatsDataSource(
        delegate=PybaseballDataSource(),
        cache=cache,
    )
    statcast_source = CachedStatcastDataSource(
        delegate=PybaseballStatcastDataSource(),
        cache=cache,
    )
    batted_ball_source = CachedBattedBallDataSource(
        delegate=PybaseballBattedBallDataSource(),
        cache=cache,
    )
    id_mapper = build_cached_sfbb_mapper(
        cache=cache,
        cache_key="ml_training",
        ttl=7 * 86400,
    )
    skill_source = CachedSkillDataSource(
        CompositeSkillDataSource(
            FanGraphsSkillDataSource(),
            StatcastSprintSpeedSource(),
            id_mapper,
        ),
        cache,
    )

    # Create trainer
    trainer = ResidualModelTrainer(
        pipeline=proj_pipeline,
        data_source=data_source,
        statcast_source=statcast_source,
        batted_ball_source=batted_ball_source,
        skill_data_source=skill_source,
        id_mapper=id_mapper,
    )

    # Train models
    model_store = ModelStore()

    typer.echo("Training batter models...")
    batter_models = trainer.train_batter_models(target_years)
    model_store.save(batter_models, name)
    typer.echo(f"  Trained stats: {batter_models.get_stats()}")

    typer.echo("Training pitcher models...")
    pitcher_models = trainer.train_pitcher_models(target_years)
    model_store.save(pitcher_models, name)
    typer.echo(f"  Trained stats: {pitcher_models.get_stats()}")

    typer.echo(f"Models saved as '{name}'")


@ml_app.command(name="list")
def list_cmd() -> None:
    """List all trained models."""
    from fantasy_baseball_manager.ml.persistence import ModelStore

    store = ModelStore()
    models = store.list_models()

    if not models:
        console.print("No trained models found.")
        return

    table = Table(title="Trained Models")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Training Years")
    table.add_column("Stats")
    table.add_column("Created")

    for meta in models:
        years_str = ", ".join(str(y) for y in meta.training_years)
        stats_str = ", ".join(meta.stats)
        table.add_row(
            meta.name,
            meta.player_type,
            years_str,
            stats_str,
            meta.created_at,
        )

    console.print(table)


@ml_app.command(name="delete")
def delete_cmd(
    name: Annotated[
        str,
        typer.Argument(help="Name of the model to delete"),
    ],
    player_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Player type: 'batter', 'pitcher', or 'all'",
        ),
    ] = "all",
) -> None:
    """Delete trained models."""
    from fantasy_baseball_manager.ml.persistence import ModelStore

    store = ModelStore()

    if player_type == "all":
        deleted_batter = store.delete(name, "batter")
        deleted_pitcher = store.delete(name, "pitcher")
        if deleted_batter or deleted_pitcher:
            typer.echo(f"Deleted model '{name}'")
        else:
            typer.echo(f"Model '{name}' not found")
    else:
        if store.delete(name, player_type):
            typer.echo(f"Deleted {player_type} model '{name}'")
        else:
            typer.echo(f"Model '{name}' ({player_type}) not found")


@ml_app.command(name="info")
def info_cmd(
    name: Annotated[
        str,
        typer.Argument(help="Name of the model"),
    ],
    player_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Player type: 'batter' or 'pitcher'",
        ),
    ] = "batter",
) -> None:
    """Show detailed information about a trained model."""
    from fantasy_baseball_manager.ml.persistence import ModelStore

    store = ModelStore()
    meta = store.get_metadata(name, player_type)

    if meta is None:
        console.print(f"Model '{name}' ({player_type}) not found")
        raise typer.Exit(1)

    console.print(f"[bold]Model:[/bold] {meta.name} ({meta.player_type})")
    console.print(f"[bold]Training years:[/bold] {', '.join(str(y) for y in meta.training_years)}")
    console.print(f"[bold]Stats:[/bold] {', '.join(meta.stats)}")
    console.print(f"[bold]Features:[/bold] {', '.join(meta.feature_names)}")
    console.print(f"[bold]Created:[/bold] {meta.created_at}")

    # Load model to show feature importances
    model_set = store.load(name, player_type)
    console.print()

    for stat in model_set.get_stats():
        model = model_set.models[stat]
        importances = model.feature_importances()
        # Sort by importance
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        table = Table(title=f"{stat} Feature Importances (top 5)")
        table.add_column("Feature")
        table.add_column("Importance", justify="right")

        for feat, imp in sorted_imp[:5]:
            table.add_row(feat, f"{imp:.4f}")

        console.print(table)
