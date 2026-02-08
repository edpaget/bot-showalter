"""CLI commands for contextual event embedding model."""

from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.console import Console

logger = logging.getLogger(__name__)

console = Console()

contextual_app = typer.Typer(help="Contextual event embedding model commands.")


@contextual_app.command(name="pretrain")
def pretrain_cmd(
    seasons: Annotated[
        str,
        typer.Option(
            "--seasons",
            "-s",
            help="Comma-separated training seasons (e.g., 2015,2016,...,2022)",
        ),
    ] = "2015,2016,2017,2018,2019,2020,2021,2022",
    val_seasons: Annotated[
        str,
        typer.Option(
            "--val-seasons",
            help="Comma-separated validation seasons (e.g., 2023)",
        ),
    ] = "2023",
    name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="Name prefix for checkpoints",
        ),
    ] = "pretrain",
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of training epochs",
        ),
    ] = 30,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Training batch size",
        ),
    ] = 32,
    learning_rate: Annotated[
        float,
        typer.Option(
            "--learning-rate",
            "--lr",
            help="Learning rate",
        ),
    ] = 1e-4,
    resume_from: Annotated[
        str | None,
        typer.Option(
            "--resume-from",
            help="Checkpoint name to resume from",
        ),
    ] = None,
    perspectives: Annotated[
        str,
        typer.Option(
            "--perspectives",
            help="Comma-separated perspectives (batter,pitcher)",
        ),
    ] = "batter,pitcher",
    max_seq_len: Annotated[
        int,
        typer.Option(
            "--max-seq-len",
            help="Maximum sequence length (caps attention memory usage)",
        ),
    ] = 512,
    amp: Annotated[
        bool,
        typer.Option(
            "--amp/--no-amp",
            help="Enable Automatic Mixed Precision (default: auto-detect CUDA)",
        ),
    ] = True,
) -> None:
    """Pre-train the contextual model using Masked Gamestate Modeling.

    Example:
        uv run python -m fantasy_baseball_manager contextual pretrain --seasons 2015,2016,2017,2018,2019,2020,2021,2022 --val-seasons 2023
    """
    import torch

    from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
    from fantasy_baseball_manager.contextual.data.vocab import (
        BB_TYPE_VOCAB,
        HANDEDNESS_VOCAB,
        PA_EVENT_VOCAB,
        PITCH_RESULT_VOCAB,
        PITCH_TYPE_VOCAB,
    )
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.heads import MaskedGamestateHead
    from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
    from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
    from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
    from fantasy_baseball_manager.contextual.training.config import PreTrainingConfig
    from fantasy_baseball_manager.contextual.training.dataset import (
        MGMDataset,
        build_player_contexts,
    )
    from fantasy_baseball_manager.contextual.training.pretrain import MGMTrainer
    from fantasy_baseball_manager.statcast.models import DEFAULT_DATA_DIR
    from fantasy_baseball_manager.statcast.store import StatcastStore

    # Parse arguments
    train_seasons = tuple(int(s.strip()) for s in seasons.split(","))
    val_season_list = tuple(int(s.strip()) for s in val_seasons.split(","))
    perspective_list = tuple(p.strip() for p in perspectives.split(","))

    # Device selection (needed to resolve AMP)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Resolve AMP: only enable on CUDA even if --amp was passed
    amp_enabled = amp and device.type == "cuda"

    config = PreTrainingConfig(
        train_seasons=train_seasons,
        val_seasons=val_season_list,
        perspectives=perspective_list,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        amp_enabled=amp_enabled,
    )

    console.print("[bold]Contextual Pre-Training (MGM)[/bold]")
    console.print(f"  Train seasons: {train_seasons}")
    console.print(f"  Val seasons:   {val_season_list}")
    console.print(f"  Perspectives:  {perspective_list}")
    console.print(f"  Epochs:        {epochs}")
    console.print(f"  Batch size:    {batch_size}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Max seq len:   {max_seq_len}")
    console.print(f"  AMP:           {amp_enabled}")

    # Build data
    store = StatcastStore(data_dir=DEFAULT_DATA_DIR)
    builder = GameSequenceBuilder(store)

    console.print("\nBuilding training data...")
    train_contexts = build_player_contexts(
        builder, config.train_seasons, config.perspectives, config.min_pitch_count,
    )
    console.print(f"  {len(train_contexts)} training player contexts")

    console.print("Building validation data...")
    val_contexts = build_player_contexts(
        builder, config.val_seasons, config.perspectives, config.min_pitch_count,
    )
    console.print(f"  {len(val_contexts)} validation player contexts")

    # Tensorize
    model_config = ModelConfig(max_seq_len=max_seq_len)
    tensorizer = Tensorizer(
        config=model_config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )

    console.print("Tensorizing sequences...")
    train_sequences = [tensorizer.tensorize_context(ctx) for ctx in train_contexts]
    val_sequences = [tensorizer.tensorize_context(ctx) for ctx in val_contexts]

    train_dataset = MGMDataset(
        sequences=train_sequences,
        config=config,
        pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
        pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
    )
    val_dataset = MGMDataset(
        sequences=val_sequences,
        config=config,
        pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
        pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
    )

    console.print(f"  {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    # Build model
    console.print(f"  Device: {device}")

    head = MaskedGamestateHead(model_config)
    model = ContextualPerformanceModel(model_config, head)
    model_store = ContextualModelStore()

    trainer = MGMTrainer(model, model_config, config, model_store, device)

    console.print("\nStarting pre-training...")
    result = trainer.train(train_dataset, val_dataset, resume_from=resume_from)

    console.print("\n[bold]Training complete![/bold]")
    console.print(f"  Val loss:              {result['val_loss']:.4f}")
    console.print(f"  Val pitch type acc:    {result['val_pitch_type_accuracy']:.4f}")
    console.print(f"  Val pitch result acc:  {result['val_pitch_result_accuracy']:.4f}")


@contextual_app.command(name="finetune")
def finetune_cmd(
    base_model: Annotated[
        str,
        typer.Option(
            "--base-model",
            help="Pre-trained checkpoint to fine-tune from",
        ),
    ] = "pretrain_best",
    perspective: Annotated[
        str,
        typer.Option(
            "--perspective",
            "-p",
            help="Player perspective: 'batter' or 'pitcher'",
        ),
    ] = "pitcher",
    context_window: Annotated[
        int,
        typer.Option(
            "--context-window",
            help="Number of prior games as context",
        ),
    ] = 10,
    seasons: Annotated[
        str,
        typer.Option(
            "--seasons",
            "-s",
            help="Comma-separated training seasons",
        ),
    ] = "2015,2016,2017,2018,2019,2020,2021,2022",
    val_seasons: Annotated[
        str,
        typer.Option(
            "--val-seasons",
            help="Comma-separated validation seasons",
        ),
    ] = "2023",
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of fine-tuning epochs",
        ),
    ] = 30,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="Training batch size",
        ),
    ] = 32,
    head_lr: Annotated[
        float,
        typer.Option(
            "--head-lr",
            help="Learning rate for prediction head",
        ),
    ] = 1e-3,
    backbone_lr: Annotated[
        float,
        typer.Option(
            "--backbone-lr",
            help="Learning rate for backbone (embedder + transformer)",
        ),
    ] = 1e-5,
    freeze_backbone: Annotated[
        bool,
        typer.Option(
            "--freeze-backbone",
            help="Freeze backbone parameters during fine-tuning",
        ),
    ] = False,
    resume_from: Annotated[
        str | None,
        typer.Option(
            "--resume-from",
            help="Fine-tune checkpoint name to resume from",
        ),
    ] = None,
    max_seq_len: Annotated[
        int,
        typer.Option(
            "--max-seq-len",
            help="Maximum sequence length",
        ),
    ] = 512,
) -> None:
    """Fine-tune a pre-trained contextual model for per-game stat prediction.

    Example:
        uv run python -m fantasy_baseball_manager contextual finetune --perspective pitcher --base-model pretrain_best
    """
    import torch

    from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
    from fantasy_baseball_manager.contextual.data.vocab import (
        BB_TYPE_VOCAB,
        HANDEDNESS_VOCAB,
        PA_EVENT_VOCAB,
        PITCH_RESULT_VOCAB,
        PITCH_TYPE_VOCAB,
    )
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.heads import PerformancePredictionHead
    from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
    from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
    from fantasy_baseball_manager.contextual.training.config import (
        BATTER_TARGET_STATS,
        PITCHER_TARGET_STATS,
        FineTuneConfig,
    )
    from fantasy_baseball_manager.contextual.training.dataset import (
        FineTuneDataset,
        build_finetune_windows,
        build_player_contexts,
    )
    from fantasy_baseball_manager.contextual.training.finetune import FineTuneTrainer
    from fantasy_baseball_manager.statcast.models import DEFAULT_DATA_DIR
    from fantasy_baseball_manager.statcast.store import StatcastStore

    # Parse arguments
    train_seasons = tuple(int(s.strip()) for s in seasons.split(","))
    val_season_list = tuple(int(s.strip()) for s in val_seasons.split(","))

    target_stats = BATTER_TARGET_STATS if perspective == "batter" else PITCHER_TARGET_STATS
    n_targets = len(target_stats)

    config = FineTuneConfig(
        train_seasons=train_seasons,
        val_seasons=val_season_list,
        perspective=perspective,
        context_window=context_window,
        epochs=epochs,
        batch_size=batch_size,
        head_learning_rate=head_lr,
        backbone_learning_rate=backbone_lr,
        freeze_backbone=freeze_backbone,
    )

    console.print("[bold]Contextual Fine-Tuning[/bold]")
    console.print(f"  Base model:    {base_model}")
    console.print(f"  Perspective:   {perspective}")
    console.print(f"  Target stats:  {target_stats}")
    console.print(f"  Context window: {context_window}")
    console.print(f"  Train seasons: {train_seasons}")
    console.print(f"  Val seasons:   {val_season_list}")
    console.print(f"  Epochs:        {epochs}")
    console.print(f"  Batch size:    {batch_size}")
    console.print(f"  Head LR:       {head_lr}")
    console.print(f"  Backbone LR:   {backbone_lr}")
    console.print(f"  Freeze backbone: {freeze_backbone}")
    console.print(f"  Max seq len:   {max_seq_len}")

    # Build model config and load pre-trained model
    model_config = ModelConfig(max_seq_len=max_seq_len)
    model_store = ContextualModelStore()

    console.print(f"\nLoading pre-trained model '{base_model}'...")
    model = model_store.load_model(base_model, model_config)

    # Swap head for fine-tuning
    head = PerformancePredictionHead(model_config, n_targets)
    model.swap_head(head)
    console.print(f"  Swapped head to PerformancePredictionHead (n_targets={n_targets})")

    # Build data
    store = StatcastStore(data_dir=DEFAULT_DATA_DIR)
    builder = GameSequenceBuilder(store)

    tensorizer = Tensorizer(
        config=model_config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )

    console.print("\nBuilding training data...")
    train_contexts = build_player_contexts(
        builder, config.train_seasons, (perspective,), min_pitch_count=10,
    )
    console.print(f"  {len(train_contexts)} training player contexts")

    console.print("Building validation data...")
    val_contexts = build_player_contexts(
        builder, config.val_seasons, (perspective,), min_pitch_count=10,
    )
    console.print(f"  {len(val_contexts)} validation player contexts")

    console.print("Building sliding windows...")
    train_windows = build_finetune_windows(train_contexts, tensorizer, config, target_stats)
    val_windows = build_finetune_windows(val_contexts, tensorizer, config, target_stats)

    train_dataset = FineTuneDataset(train_windows)
    val_dataset = FineTuneDataset(val_windows)
    console.print(f"  {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    console.print(f"  Device: {device}")

    # Train
    trainer = FineTuneTrainer(model, model_config, config, model_store, target_stats, device)

    console.print("\nStarting fine-tuning...")
    result = trainer.train(train_dataset, val_dataset, resume_from=resume_from)

    console.print("\n[bold]Fine-tuning complete![/bold]")
    console.print(f"  Val loss: {result['val_loss']:.4f}")
    for stat in target_stats:
        mse_key = f"val_{stat}_mse"
        mae_key = f"val_{stat}_mae"
        if mse_key in result:
            console.print(f"  {stat}: MSE={result[mse_key]:.4f}  MAE={result[mae_key]:.4f}")
