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

    config = PreTrainingConfig(
        train_seasons=train_seasons,
        val_seasons=val_season_list,
        perspectives=perspective_list,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    console.print("[bold]Contextual Pre-Training (MGM)[/bold]")
    console.print(f"  Train seasons: {train_seasons}")
    console.print(f"  Val seasons:   {val_season_list}")
    console.print(f"  Perspectives:  {perspective_list}")
    console.print(f"  Epochs:        {epochs}")
    console.print(f"  Batch size:    {batch_size}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Max seq len:   {max_seq_len}")

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
