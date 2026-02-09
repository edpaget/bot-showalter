"""CLI commands for contextual event embedding model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from fantasy_baseball_manager.registry.registry import ModelRegistry

import typer
from rich.console import Console

logger = logging.getLogger(__name__)

console = Console()

contextual_app = typer.Typer(help="Contextual event embedding model commands.")


def _get_registry() -> ModelRegistry:
    """Create the model registry for CLI commands."""
    from fantasy_baseball_manager.registry.factory import create_model_registry

    return create_model_registry()


def _build_pretrain_meta(
    seasons: tuple[int, ...],
    val_seasons: tuple[int, ...],
    perspectives: tuple[str, ...],
    max_seq_len: int,
    min_pitch_count: int,
) -> dict[str, object]:
    return {
        "seasons": sorted(seasons),
        "val_seasons": sorted(val_seasons),
        "perspectives": sorted(perspectives),
        "max_seq_len": max_seq_len,
        "min_pitch_count": min_pitch_count,
    }


def _build_finetune_meta(
    seasons: tuple[int, ...],
    val_seasons: tuple[int, ...],
    perspective: str,
    context_window: int,
    max_seq_len: int,
    min_games: int,
) -> dict[str, object]:
    return {
        "seasons": sorted(seasons),
        "val_seasons": sorted(val_seasons),
        "perspective": perspective,
        "context_window": context_window,
        "max_seq_len": max_seq_len,
        "min_games": min_games,
    }


def _log_meta_mismatch(
    stored: dict[str, object] | None,
    expected: dict[str, object],
) -> None:
    """Print which metadata keys differ between stored and expected."""
    if stored is None:
        console.print("  [yellow]No metadata file found[/yellow]")
        return
    all_keys = sorted(set(stored) | set(expected))
    for key in all_keys:
        s_val = stored.get(key)
        e_val = expected.get(key)
        if s_val != e_val:
            console.print(f"  [yellow]  {key}: stored={s_val!r}  expected={e_val!r}[/yellow]")


@contextual_app.command(name="prepare-data")
def prepare_data_cmd(
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="What to prepare: 'pretrain', 'finetune', or 'all'",
        ),
    ] = "all",
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
    perspectives: Annotated[
        str,
        typer.Option(
            "--perspectives",
            help="Comma-separated perspectives for pretrain (batter,pitcher)",
        ),
    ] = "batter,pitcher",
    perspective: Annotated[
        str,
        typer.Option(
            "--perspective",
            "-p",
            help="Player perspective for finetune: 'batter' or 'pitcher'",
        ),
    ] = "pitcher",
    context_window: Annotated[
        int | None,
        typer.Option(
            "--context-window",
            help="Number of prior games as context (finetune). Defaults to 30 for batter, 10 for pitcher.",
        ),
    ] = None,
    max_seq_len: Annotated[
        int,
        typer.Option(
            "--max-seq-len",
            help="Maximum sequence length",
        ),
    ] = 512,
    min_pitch_count: Annotated[
        int,
        typer.Option(
            "--min-pitch-count",
            help="Minimum pitch count to include a player context (pretrain)",
        ),
    ] = 10,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Max parallel workers for data building (default: CPU count)",
        ),
    ] = None,
) -> None:
    """Pre-build tensorized sequences and save to disk.

    Saves prepared data so that pretrain/finetune can skip expensive data
    building steps (parquet I/O, game sequence building, tensorization).

    Example:
        uv run python -m fantasy_baseball_manager contextual prepare-data --mode all
    """
    from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
    from fantasy_baseball_manager.contextual.data.vocab import (
        BB_TYPE_VOCAB,
        HANDEDNESS_VOCAB,
        PA_EVENT_VOCAB,
        PITCH_RESULT_VOCAB,
        PITCH_TYPE_VOCAB,
    )
    from fantasy_baseball_manager.contextual.data_store import PreparedDataStore
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
    from fantasy_baseball_manager.contextual.training.config import (
        BATTER_TARGET_STATS,
        DEFAULT_BATTER_CONTEXT_WINDOW,
        DEFAULT_PITCHER_CONTEXT_WINDOW,
        PITCHER_TARGET_STATS,
        FineTuneConfig,
    )
    from fantasy_baseball_manager.contextual.training.dataset import (
        build_finetune_windows,
        build_player_contexts,
        tensorize_contexts,
    )
    from fantasy_baseball_manager.statcast.models import DEFAULT_DATA_DIR
    from fantasy_baseball_manager.statcast.store import StatcastStore

    train_seasons = tuple(int(s.strip()) for s in seasons.split(","))
    val_season_list = tuple(int(s.strip()) for s in val_seasons.split(","))
    perspective_list = tuple(p.strip() for p in perspectives.split(","))

    if context_window is None:
        context_window = DEFAULT_BATTER_CONTEXT_WINDOW if perspective == "batter" else DEFAULT_PITCHER_CONTEXT_WINDOW

    model_config = ModelConfig(max_seq_len=max_seq_len)
    tensorizer = Tensorizer(
        config=model_config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )

    store = StatcastStore(data_dir=DEFAULT_DATA_DIR)
    builder = GameSequenceBuilder(store)
    data_store = PreparedDataStore()

    do_pretrain = mode in ("pretrain", "all")
    do_finetune = mode in ("finetune", "all")

    if do_pretrain:
        console.print("[bold]Preparing pre-training data...[/bold]")

        console.print("  Building training contexts...")
        train_contexts = build_player_contexts(
            builder, train_seasons, perspective_list, min_pitch_count,
            max_workers=workers,
        )
        console.print(f"  {len(train_contexts)} training player contexts")

        console.print("  Building validation contexts...")
        val_contexts = build_player_contexts(
            builder, val_season_list, perspective_list, min_pitch_count,
            max_workers=workers,
        )
        console.print(f"  {len(val_contexts)} validation player contexts")

        console.print("  Tensorizing training sequences...")
        train_sequences = tensorize_contexts(tensorizer, train_contexts, max_workers=workers)
        console.print("  Tensorizing validation sequences...")
        val_sequences = tensorize_contexts(tensorizer, val_contexts, max_workers=workers)

        meta = _build_pretrain_meta(
            train_seasons, val_season_list, perspective_list, max_seq_len, min_pitch_count,
        )
        data_store.save_pretrain_data("pretrain_train", train_sequences, meta)
        data_store.save_pretrain_data("pretrain_val", val_sequences, meta)

        console.print(
            f"  Saved {len(train_sequences)} train + {len(val_sequences)} val sequences"
        )

    if do_finetune:
        # When mode is "all", prepare both perspectives; otherwise just the one requested
        ft_perspectives = ("batter", "pitcher") if mode == "all" else (perspective,)

        for ft_perspective in ft_perspectives:
            ft_context_window = (
                DEFAULT_BATTER_CONTEXT_WINDOW if ft_perspective == "batter"
                else DEFAULT_PITCHER_CONTEXT_WINDOW
            ) if mode == "all" else context_window

            console.print(f"[bold]Preparing fine-tune data ({ft_perspective})...[/bold]")

            target_stats = BATTER_TARGET_STATS if ft_perspective == "batter" else PITCHER_TARGET_STATS
            ft_config = FineTuneConfig(
                train_seasons=train_seasons,
                val_seasons=val_season_list,
                perspective=ft_perspective,
                context_window=ft_context_window,
                min_games=ft_context_window + 5,
            )

            console.print("  Building training contexts...")
            train_contexts = build_player_contexts(
                builder, train_seasons, (ft_perspective,), min_pitch_count=10,
                max_workers=workers,
            )
            console.print(f"  {len(train_contexts)} training player contexts")

            console.print("  Building validation contexts...")
            val_contexts = build_player_contexts(
                builder, val_season_list, (ft_perspective,), min_pitch_count=10,
                max_workers=workers,
            )
            console.print(f"  {len(val_contexts)} validation player contexts")

            console.print("  Building sliding windows...")
            train_windows = build_finetune_windows(
                train_contexts, tensorizer, ft_config, target_stats, max_workers=workers,
            )
            val_windows = build_finetune_windows(
                val_contexts, tensorizer, ft_config, target_stats, max_workers=workers,
            )

            ft_meta = _build_finetune_meta(
                train_seasons, val_season_list, ft_perspective, ft_context_window,
                max_seq_len, ft_config.min_games,
            )
            train_name = f"finetune_{ft_perspective}_train"
            val_name = f"finetune_{ft_perspective}_val"
            data_store.save_finetune_data(train_name, train_windows, ft_meta)
            data_store.save_finetune_data(val_name, val_windows, ft_meta)

            console.print(
                f"  Saved {len(train_windows)} train + {len(val_windows)} val windows"
            )

    console.print("[bold green]Done![/bold green]")


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
    prepared_data: Annotated[
        bool,
        typer.Option(
            "--prepared-data/--no-prepared-data",
            help="Load pre-built tensorized data if available",
        ),
    ] = True,
) -> None:
    """Pre-train the contextual model using Masked Gamestate Modeling.

    Example:
        uv run python -m fantasy_baseball_manager contextual pretrain --seasons 2015,2016,2017,2018,2019,2020,2021,2022 --val-seasons 2023
    """
    import torch

    from fantasy_baseball_manager.contextual.data.vocab import (
        PITCH_RESULT_VOCAB,
        PITCH_TYPE_VOCAB,
    )
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.heads import MaskedGamestateHead
    from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
    from fantasy_baseball_manager.contextual.training.config import PreTrainingConfig
    from fantasy_baseball_manager.contextual.training.dataset import MGMDataset
    from fantasy_baseball_manager.contextual.training.pretrain import MGMTrainer

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

    # Try loading prepared data
    train_sequences = None
    val_sequences = None

    if prepared_data:
        from fantasy_baseball_manager.contextual.data_store import PreparedDataStore

        data_store = PreparedDataStore()
        if data_store.exists("pretrain_train") and data_store.exists("pretrain_val"):
            expected_meta = _build_pretrain_meta(
                train_seasons, val_season_list, perspective_list,
                max_seq_len, config.min_pitch_count,
            )
            stored_meta = data_store.load_meta("pretrain_train")
            if stored_meta == expected_meta:
                console.print("\nLoading prepared data...")
                train_sequences = data_store.load_pretrain_data("pretrain_train")
                val_sequences = data_store.load_pretrain_data("pretrain_val")
                console.print(
                    f"  Loaded prepared data ({len(train_sequences)} train, "
                    f"{len(val_sequences)} val sequences)"
                )
            else:
                console.print("[red]Prepared data exists but parameters don't match:[/red]")
                _log_meta_mismatch(stored_meta, expected_meta)
                raise SystemExit(
                    "Re-run 'prepare-data' with matching parameters, or use --no-prepared-data to skip."
                )

    if train_sequences is None or val_sequences is None:
        from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
        from fantasy_baseball_manager.contextual.data.vocab import (
            BB_TYPE_VOCAB,
            HANDEDNESS_VOCAB,
            PA_EVENT_VOCAB,
        )
        from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
        from fantasy_baseball_manager.contextual.training.dataset import build_player_contexts
        from fantasy_baseball_manager.statcast.models import DEFAULT_DATA_DIR
        from fantasy_baseball_manager.statcast.store import StatcastStore

        model_config_for_tensorizer = ModelConfig(max_seq_len=max_seq_len)
        tensorizer = Tensorizer(
            config=model_config_for_tensorizer,
            pitch_type_vocab=PITCH_TYPE_VOCAB,
            pitch_result_vocab=PITCH_RESULT_VOCAB,
            bb_type_vocab=BB_TYPE_VOCAB,
            handedness_vocab=HANDEDNESS_VOCAB,
            pa_event_vocab=PA_EVENT_VOCAB,
        )

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

        console.print("Tensorizing sequences...")
        train_sequences = [tensorizer.tensorize_context(ctx) for ctx in train_contexts]
        val_sequences = [tensorizer.tensorize_context(ctx) for ctx in val_contexts]

    model_config = ModelConfig(max_seq_len=max_seq_len)

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
    registry = _get_registry()
    model_store = registry.contextual_store

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
        int | None,
        typer.Option(
            "--context-window",
            help="Number of prior games as context. Defaults to 30 for batter, 10 for pitcher.",
        ),
    ] = None,
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
        int | None,
        typer.Option(
            "--max-seq-len",
            help="Maximum sequence length (auto-detected from pretrained model if omitted)",
        ),
    ] = None,
    prepared_data: Annotated[
        bool,
        typer.Option(
            "--prepared-data/--no-prepared-data",
            help="Load pre-built tensorized data if available",
        ),
    ] = True,
) -> None:
    """Fine-tune a pre-trained contextual model for per-game stat prediction.

    Example:
        uv run python -m fantasy_baseball_manager contextual finetune --perspective pitcher --base-model pretrain_best
    """
    import torch

    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.heads import PerformancePredictionHead
    from fantasy_baseball_manager.contextual.training.config import (
        BATTER_TARGET_STATS,
        DEFAULT_BATTER_CONTEXT_WINDOW,
        DEFAULT_PITCHER_CONTEXT_WINDOW,
        PITCHER_TARGET_STATS,
        FineTuneConfig,
    )
    from fantasy_baseball_manager.contextual.training.dataset import FineTuneDataset
    from fantasy_baseball_manager.contextual.training.finetune import FineTuneTrainer

    # Parse arguments
    train_seasons = tuple(int(s.strip()) for s in seasons.split(","))
    val_season_list = tuple(int(s.strip()) for s in val_seasons.split(","))

    if context_window is None:
        context_window = DEFAULT_BATTER_CONTEXT_WINDOW if perspective == "batter" else DEFAULT_PITCHER_CONTEXT_WINDOW

    target_stats = BATTER_TARGET_STATS if perspective == "batter" else PITCHER_TARGET_STATS
    n_targets = len(target_stats)

    # Auto-detect max_seq_len from pretrained model
    registry = _get_registry()
    model_store = registry.contextual_store
    pretrain_state = torch.load(model_store._model_path(base_model), weights_only=True)
    pretrain_seq_len: int = pretrain_state["positional_encoding.pe"].shape[1]
    if max_seq_len is None:
        max_seq_len = pretrain_seq_len
        console.print(f"  Auto-detected max_seq_len={max_seq_len} from pretrained model")
    elif max_seq_len != pretrain_seq_len:
        raise SystemExit(
            f"--max-seq-len {max_seq_len} does not match pretrained model "
            f"positional encoding size {pretrain_seq_len}. "
            f"Either omit --max-seq-len to auto-detect or retrain with the desired length."
        )

    config = FineTuneConfig(
        train_seasons=train_seasons,
        val_seasons=val_season_list,
        perspective=perspective,
        context_window=context_window,
        min_games=context_window + 5,
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

    console.print(f"\nLoading pre-trained model '{base_model}'...")
    model = model_store.load_model(base_model, model_config)

    # Swap head for fine-tuning
    head = PerformancePredictionHead(model_config, n_targets)
    model.swap_head(head)
    console.print(f"  Swapped head to PerformancePredictionHead (n_targets={n_targets})")

    # Try loading prepared data
    train_windows = None
    val_windows = None

    if prepared_data:
        from fantasy_baseball_manager.contextual.data_store import PreparedDataStore

        data_store = PreparedDataStore()
        train_name = f"finetune_{perspective}_train"
        val_name = f"finetune_{perspective}_val"

        if data_store.exists(train_name) and data_store.exists(val_name):
            expected_meta = _build_finetune_meta(
                train_seasons, val_season_list, perspective, context_window,
                max_seq_len, config.min_games,
            )
            stored_meta = data_store.load_meta(train_name)
            if stored_meta == expected_meta:
                console.print("\nLoading prepared data...")
                train_windows = data_store.load_finetune_data(train_name)
                val_windows = data_store.load_finetune_data(val_name)
                console.print(
                    f"  Loaded prepared data ({len(train_windows)} train, "
                    f"{len(val_windows)} val windows)"
                )
            else:
                console.print("[red]Prepared data exists but parameters don't match:[/red]")
                _log_meta_mismatch(stored_meta, expected_meta)
                raise SystemExit(
                    "Re-run 'prepare-data' with matching parameters, or use --no-prepared-data to skip."
                )

    if train_windows is None or val_windows is None:
        from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
        from fantasy_baseball_manager.contextual.data.vocab import (
            BB_TYPE_VOCAB,
            HANDEDNESS_VOCAB,
            PA_EVENT_VOCAB,
            PITCH_RESULT_VOCAB,
            PITCH_TYPE_VOCAB,
        )
        from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
        from fantasy_baseball_manager.contextual.training.dataset import (
            build_finetune_windows,
            build_player_contexts,
        )
        from fantasy_baseball_manager.statcast.models import DEFAULT_DATA_DIR
        from fantasy_baseball_manager.statcast.store import StatcastStore

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
        bl_mse_key = f"baseline_{stat}_mse"
        bl_mae_key = f"baseline_{stat}_mae"
        if mse_key in result:
            parts = f"  {stat}: MSE={result[mse_key]:.4f}  MAE={result[mae_key]:.4f}"
            if bl_mse_key in result:
                parts += f"  (baseline MSE={result[bl_mse_key]:.4f}  MAE={result[bl_mae_key]:.4f})"
            console.print(parts)
