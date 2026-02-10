"""CLI commands for ML model training and management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from fantasy_baseball_manager.registry.registry import ModelRegistry

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

console = Console()

ml_app = typer.Typer(help="Machine learning model commands.")


def _get_registry() -> ModelRegistry:
    """Create the model registry for CLI commands."""
    from fantasy_baseball_manager.registry.factory import create_model_registry

    return create_model_registry()


def resolve_version(
    registry: ModelRegistry,
    base_name: str,
    model_type: str,
    version: int | None,
) -> tuple[int, str]:
    """Resolve the version number and versioned name for a training run.

    When *version* is ``None``, auto-increments by computing the max of
    ``next_version`` across batter and pitcher so that both player types
    share a consistent version number.

    Returns:
        ``(version, versioned_name)`` - e.g. ``(1, "default")`` or
        ``(2, "default_v2")``.
    """
    if version is None:
        batter_next = registry.next_version(base_name, model_type, "batter")
        pitcher_next = registry.next_version(base_name, model_type, "pitcher")
        version = max(batter_next, pitcher_next)
    if version == 1:
        return version, base_name
    return version, f"{base_name}_v{version}"


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
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Run time-series holdout validation and save results with model",
        ),
    ] = False,
    version: Annotated[
        int | None,
        typer.Option(
            "--version",
            help="Model version number. Auto-increments if omitted.",
        ),
    ] = None,
) -> None:
    """Train gradient boosting residual models on historical data.

    Example:
        uv run python -m fantasy_baseball_manager ml train --years 2020,2021,2022,2023 --name default
        uv run python -m fantasy_baseball_manager ml train --years 2020,2021,2022,2023 --validate
    """
    from fantasy_baseball_manager.cache.factory import create_cache_store
    from fantasy_baseball_manager.cache.serialization import DataclassListSerializer
    from fantasy_baseball_manager.cache.wrapper import cached
    from fantasy_baseball_manager.marcel.data_source import (
        create_batting_source,
        create_pitching_source,
        create_team_batting_source,
        create_team_pitching_source,
    )
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
    from fantasy_baseball_manager.ml.persistence import ModelStore
    from fantasy_baseball_manager.ml.training import ResidualModelTrainer
    from fantasy_baseball_manager.ml.validation import TimeSeriesHoldout, ValidationReport
    from fantasy_baseball_manager.pipeline.batted_ball_data import (
        CachedBattedBallDataSource,
        PybaseballBattedBallDataSource,
    )
    from fantasy_baseball_manager.pipeline.feature_store import FeatureStore
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
    stats_ttl = 30 * 86400  # 30 days
    batting_source = cached(
        create_batting_source(),
        namespace="stats_batting",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(BattingSeasonStats),
    )
    team_batting_source = cached(
        create_team_batting_source(),
        namespace="stats_team_batting",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(BattingSeasonStats),
    )
    pitching_source = cached(
        create_pitching_source(),
        namespace="stats_pitching",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(PitchingSeasonStats),
    )
    team_pitching_source = cached(
        create_team_pitching_source(),
        namespace="stats_team_pitching",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(PitchingSeasonStats),
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

    # Create feature store and trainer
    feature_store = FeatureStore(
        statcast_source=statcast_source,
        batted_ball_source=batted_ball_source,
        skill_data_source=skill_source,
    )
    trainer = ResidualModelTrainer(
        pipeline=proj_pipeline,
        batting_source=batting_source,
        team_batting_source=team_batting_source,
        pitching_source=pitching_source,
        team_pitching_source=team_pitching_source,
        statcast_source=statcast_source,
        batted_ball_source=batted_ball_source,
        skill_data_source=skill_source,
        id_mapper=id_mapper,
        feature_store=feature_store,
    )

    # Train models
    registry = _get_registry()
    resolved_version, versioned_name = resolve_version(registry, name, "gb_residual", version)
    model_store = ModelStore(model_dir=registry.gb_store.model_dir)

    # Run validation if requested
    batter_validation: ValidationReport | None = None
    pitcher_validation: ValidationReport | None = None

    if validate:
        if len(target_years) < 2:
            typer.echo("Warning: Need at least 2 years for validation. Skipping validation.")
        else:
            strategy = TimeSeriesHoldout(holdout_years=1)
            typer.echo(f"Running validation with strategy: {strategy.name}")

            typer.echo("Validating batter models...")
            batter_validation = trainer.validate_batter_models(target_years, strategy)
            _print_validation_report(batter_validation)

            typer.echo("Validating pitcher models...")
            pitcher_validation = trainer.validate_pitcher_models(target_years, strategy)
            _print_validation_report(pitcher_validation)

    typer.echo("\nTraining batter models...")
    batter_models = trainer.train_batter_models(target_years)
    model_store.save(batter_models, versioned_name, batter_validation, version=resolved_version)
    typer.echo(f"  Trained stats: {batter_models.get_stats()}")

    typer.echo("Training pitcher models...")
    pitcher_models = trainer.train_pitcher_models(target_years)
    model_store.save(pitcher_models, versioned_name, pitcher_validation, version=resolved_version)
    typer.echo(f"  Trained stats: {pitcher_models.get_stats()}")

    typer.echo(f"\nModels saved as '{versioned_name}'")


@ml_app.command(name="list")
def list_cmd(
    model_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-t",
            help="Filter by model type: gb_residual, mtl, mle, or contextual",
        ),
    ] = None,
) -> None:
    """List all trained models."""
    registry = _get_registry()
    models = registry.list_all(model_type=model_type)

    if not models:
        console.print("No trained models found.")
        return

    table = Table(title="Trained Models")
    table.add_column("Name", no_wrap=True)
    table.add_column("Model Type", no_wrap=True)
    table.add_column("Player Type", no_wrap=True)
    table.add_column("Version", justify="right")
    table.add_column("Training Years")
    table.add_column("Stats")
    table.add_column("Created")

    for meta in models:
        years_str = ", ".join(str(y) for y in meta.training_years)
        stats_str = ", ".join(meta.stats)
        table.add_row(
            meta.name,
            meta.model_type,
            meta.player_type,
            str(meta.version),
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

    registry = _get_registry()
    store = ModelStore(model_dir=registry.gb_store.model_dir)

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

    registry = _get_registry()
    store = ModelStore(model_dir=registry.gb_store.model_dir)
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


@ml_app.command(name="compare")
def compare_cmd(
    name_a: Annotated[
        str,
        typer.Argument(help="Name of the first model"),
    ],
    name_b: Annotated[
        str,
        typer.Argument(help="Name of the second model"),
    ],
    model_type: Annotated[
        str,
        typer.Option(
            "--model-type",
            "-m",
            help="Model type: gb_residual, mtl, or mle",
        ),
    ] = "gb_residual",
    player_type: Annotated[
        str,
        typer.Option(
            "--player-type",
            "-p",
            help="Player type: batter or pitcher",
        ),
    ] = "batter",
) -> None:
    """Compare two trained models side by side.

    Shows metadata, training years differences, and metric deltas.

    Example:
        uv run python -m fantasy_baseball_manager ml compare default default_v2 --model-type gb_residual
    """
    registry = _get_registry()
    try:
        comparison = registry.compare(name_a, name_b, model_type, player_type)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e

    meta_a = comparison["a"]
    meta_b = comparison["b"]

    # Metadata table
    meta_table = Table(title="Model Comparison")
    meta_table.add_column("Field", no_wrap=True)
    meta_table.add_column(name_a, no_wrap=True)
    meta_table.add_column(name_b, no_wrap=True)
    meta_table.add_row("Name", meta_a["name"], meta_b["name"])
    meta_table.add_row("Version", str(meta_a["version"]), str(meta_b["version"]))
    meta_table.add_row(
        "Training Years",
        ", ".join(str(y) for y in meta_a["training_years"]),
        ", ".join(str(y) for y in meta_b["training_years"]),
    )
    meta_table.add_row("Created", meta_a["created_at"], meta_b["created_at"])
    console.print(meta_table)

    # Training years diff
    years_diff = comparison["training_years_diff"]
    a_only = years_diff["a_only"]
    b_only = years_diff["b_only"]
    if a_only or b_only:
        console.print()
        console.print("[bold]Training Years Diff[/bold]")
        if a_only:
            console.print(f"  Only in {name_a}: {', '.join(str(y) for y in a_only)}")
        if b_only:
            console.print(f"  Only in {name_b}: {', '.join(str(y) for y in b_only)}")

    # Metrics table
    metrics_diff = comparison["metrics_diff"]
    if metrics_diff:
        console.print()
        metrics_table = Table(title="Metrics")
        metrics_table.add_column("Metric", no_wrap=True)
        metrics_table.add_column(name_a, justify="right")
        metrics_table.add_column(name_b, justify="right")
        metrics_table.add_column("Delta", justify="right")

        for key, diff in metrics_diff.items():
            val_a = diff["a"]
            val_b = diff["b"]
            a_str = f"{val_a:+.4f}" if isinstance(val_a, int | float) else str(val_a)
            b_str = f"{val_b:+.4f}" if isinstance(val_b, int | float) else str(val_b)
            delta_str = f"{diff['delta']:+.4f}" if "delta" in diff else "-"
            metrics_table.add_row(key, a_str, b_str, delta_str)

        console.print(metrics_table)


@ml_app.command(name="validate")
def validate_cmd(
    years: Annotated[
        str,
        typer.Option(
            "--years",
            "-y",
            help="Comma-separated years for validation (e.g., 2020,2021,2022,2023)",
        ),
    ],
    strategy: Annotated[
        str,
        typer.Option(
            "--strategy",
            "-s",
            help="Validation strategy: 'time_series' or 'loyo' (leave-one-year-out)",
        ),
    ] = "time_series",
    player_type: Annotated[
        str,
        typer.Option(
            "--type",
            "-t",
            help="Player type: 'batter', 'pitcher', or 'all'",
        ),
    ] = "all",
    pipeline: Annotated[
        str,
        typer.Option(
            "--pipeline",
            "-p",
            help="Base pipeline to use for generating projections (marcel or marcel_full)",
        ),
    ] = "marcel",
    holdout_years: Annotated[
        int,
        typer.Option(
            "--holdout-years",
            help="Number of holdout years for time_series strategy",
        ),
    ] = 1,
    early_stopping: Annotated[
        bool,
        typer.Option(
            "--early-stopping/--no-early-stopping",
            help="Enable early stopping during training",
        ),
    ] = False,
) -> None:
    """Validate models without saving.

    Example:
        uv run python -m fantasy_baseball_manager ml validate --years 2020,2021,2022,2023 --strategy time_series
    """
    from fantasy_baseball_manager.cache.factory import create_cache_store
    from fantasy_baseball_manager.cache.serialization import DataclassListSerializer
    from fantasy_baseball_manager.cache.wrapper import cached
    from fantasy_baseball_manager.marcel.data_source import (
        create_batting_source,
        create_pitching_source,
        create_team_batting_source,
        create_team_pitching_source,
    )
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
    from fantasy_baseball_manager.ml.training import ResidualModelTrainer
    from fantasy_baseball_manager.ml.validation import (
        EarlyStoppingConfig,
        LeaveOneYearOut,
        TimeSeriesHoldout,
    )
    from fantasy_baseball_manager.pipeline.batted_ball_data import (
        CachedBattedBallDataSource,
        PybaseballBattedBallDataSource,
    )
    from fantasy_baseball_manager.pipeline.feature_store import FeatureStore
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
    typer.echo(f"Validating models for years: {target_years}")

    # Build validation strategy
    validation_strategy: TimeSeriesHoldout | LeaveOneYearOut
    if strategy == "time_series":
        validation_strategy = TimeSeriesHoldout(holdout_years=holdout_years)
    elif strategy == "loyo":
        validation_strategy = LeaveOneYearOut()
    else:
        typer.echo(f"Unknown strategy: {strategy}. Use 'time_series' or 'loyo'.")
        raise typer.Exit(1)

    typer.echo(f"Using strategy: {validation_strategy.name}")

    # Build early stopping config
    es_config = EarlyStoppingConfig(enabled=early_stopping) if early_stopping else None

    # Build pipeline
    typer.echo(f"Using pipeline: {pipeline}")
    proj_pipeline = build_pipeline(pipeline)

    # Setup data sources
    cache = create_cache_store()
    stats_ttl = 30 * 86400  # 30 days
    batting_source = cached(
        create_batting_source(),
        namespace="stats_batting",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(BattingSeasonStats),
    )
    team_batting_source = cached(
        create_team_batting_source(),
        namespace="stats_team_batting",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(BattingSeasonStats),
    )
    pitching_source = cached(
        create_pitching_source(),
        namespace="stats_pitching",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(PitchingSeasonStats),
    )
    team_pitching_source = cached(
        create_team_pitching_source(),
        namespace="stats_team_pitching",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(PitchingSeasonStats),
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
        cache_key="ml_validation",
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

    # Create feature store and trainer
    feature_store = FeatureStore(
        statcast_source=statcast_source,
        batted_ball_source=batted_ball_source,
        skill_data_source=skill_source,
    )
    trainer = ResidualModelTrainer(
        pipeline=proj_pipeline,
        batting_source=batting_source,
        team_batting_source=team_batting_source,
        pitching_source=pitching_source,
        team_pitching_source=team_pitching_source,
        statcast_source=statcast_source,
        batted_ball_source=batted_ball_source,
        skill_data_source=skill_source,
        id_mapper=id_mapper,
        feature_store=feature_store,
    )

    # Run validation
    if player_type in ("batter", "all"):
        typer.echo("\nValidating batter models...")
        batter_report = trainer.validate_batter_models(target_years, validation_strategy, es_config)
        _print_validation_report(batter_report)

    if player_type in ("pitcher", "all"):
        typer.echo("\nValidating pitcher models...")
        pitcher_report = trainer.validate_pitcher_models(target_years, validation_strategy, es_config)
        _print_validation_report(pitcher_report)


@ml_app.command(name="train-mtl")
def train_mtl_cmd(
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
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Print validation metrics after training",
        ),
    ] = False,
    version: Annotated[
        int | None,
        typer.Option(
            "--version",
            help="Model version number. Auto-increments if omitted.",
        ),
    ] = None,
) -> None:
    """Train multi-task learning neural network models on historical data.

    MTL models predict raw stat rates directly using Statcast features.
    Unlike gradient boosting residual models, MTL can be used as a
    standalone projection system or blended with Marcel.

    Requires PyTorch: uv sync --extra mtl

    Example:
        uv run python -m fantasy_baseball_manager ml train-mtl --years 2020,2021,2022,2023 --name default
        uv run python -m fantasy_baseball_manager ml train-mtl --years 2020,2021,2022,2023 --validate
    """
    try:
        from fantasy_baseball_manager.ml.mtl.persistence import MTLModelStore
        from fantasy_baseball_manager.ml.mtl.trainer import MTLTrainer
    except ImportError as e:
        console.print("[red]Error:[/red] PyTorch is required for MTL models.")
        console.print("Install with: [bold]uv sync --extra mtl[/bold]")
        raise typer.Exit(1) from e

    from fantasy_baseball_manager.cache.factory import create_cache_store
    from fantasy_baseball_manager.cache.serialization import DataclassListSerializer
    from fantasy_baseball_manager.cache.wrapper import cached
    from fantasy_baseball_manager.marcel.data_source import (
        create_batting_source,
        create_pitching_source,
    )
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
    from fantasy_baseball_manager.pipeline.batted_ball_data import (
        CachedBattedBallDataSource,
        PybaseballBattedBallDataSource,
    )
    from fantasy_baseball_manager.pipeline.feature_store import FeatureStore
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
    typer.echo(f"Training MTL models for target years: {target_years}")

    # Setup data sources
    cache = create_cache_store()
    stats_ttl = 30 * 86400  # 30 days
    batting_source = cached(
        create_batting_source(),
        namespace="stats_batting",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(BattingSeasonStats),
    )
    pitching_source = cached(
        create_pitching_source(),
        namespace="stats_pitching",
        ttl_seconds=stats_ttl,
        serializer=DataclassListSerializer(PitchingSeasonStats),
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
        cache_key="mtl_training",
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

    # Create feature store and trainer
    feature_store = FeatureStore(
        statcast_source=statcast_source,
        batted_ball_source=batted_ball_source,
        skill_data_source=skill_source,
    )
    trainer = MTLTrainer(
        batting_source=batting_source,
        pitching_source=pitching_source,
        statcast_source=statcast_source,
        batted_ball_source=batted_ball_source,
        skill_data_source=skill_source,
        id_mapper=id_mapper,
        feature_store=feature_store,
    )

    registry = _get_registry()
    resolved_version, versioned_name = resolve_version(registry, name, "mtl", version)
    model_store = MTLModelStore(model_dir=registry.mtl_store.model_dir)

    # Train batter model
    typer.echo("\nTraining MTL batter model...")
    batter_model, batter_metrics = trainer.train_batter_model(target_years)

    if batter_model.is_fitted:
        model_store.save_batter_model(batter_model, versioned_name, version=resolved_version)
        typer.echo(f"  Trained batter model with {len(batter_model.feature_names)} features")

        if validate and batter_metrics:
            _print_mtl_validation_metrics("Batter", batter_metrics)
    else:
        typer.echo("  [yellow]Warning:[/yellow] Insufficient data for batter model")

    # Train pitcher model
    typer.echo("\nTraining MTL pitcher model...")
    pitcher_model, pitcher_metrics = trainer.train_pitcher_model(target_years)

    if pitcher_model.is_fitted:
        model_store.save_pitcher_model(pitcher_model, versioned_name, version=resolved_version)
        typer.echo(f"  Trained pitcher model with {len(pitcher_model.feature_names)} features")

        if validate and pitcher_metrics:
            _print_mtl_validation_metrics("Pitcher", pitcher_metrics)
    else:
        typer.echo("  [yellow]Warning:[/yellow] Insufficient data for pitcher model")

    typer.echo(f"\nMTL models saved as '{versioned_name}'")


def _print_mtl_validation_metrics(player_type: str, metrics: dict[str, float]) -> None:
    """Print MTL validation metrics to the console."""
    table = Table(title=f"{player_type} MTL Validation Metrics (RMSE)")
    table.add_column("Stat")
    table.add_column("RMSE", justify="right")

    for key, value in sorted(metrics.items()):
        stat = key.replace("_rmse", "")
        table.add_row(stat, f"{value:.6f}")

    console.print(table)


def _print_validation_report(report: object) -> None:
    """Print a validation report to the console."""
    from fantasy_baseball_manager.ml.validation import ValidationReport

    if not isinstance(report, ValidationReport):
        return

    console.print(f"\n[bold]{report.player_type.title()} Validation Results[/bold]")
    console.print(f"Strategy: {report.strategy_name}")
    console.print(f"Training years: {', '.join(str(y) for y in report.training_years)}")
    console.print(f"Holdout years: {', '.join(str(y) for y in report.holdout_years)}")

    table = Table(title="Validation Metrics by Stat")
    table.add_column("Stat")
    table.add_column("Mean RMSE", justify="right")
    table.add_column("Mean MAE", justify="right")
    table.add_column("Mean R²", justify="right")
    table.add_column("Samples", justify="right")

    for stat_result in report.stat_results:
        table.add_row(
            stat_result.stat_name,
            f"{stat_result.mean_rmse:.3f}",
            f"{stat_result.mean_mae:.3f}",
            f"{stat_result.mean_r_squared:.3f}",
            str(stat_result.total_samples),
        )

    console.print(table)
