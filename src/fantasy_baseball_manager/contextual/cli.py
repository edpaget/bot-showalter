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
    target_mode: str = "rates",
    target_window: int = 5,
) -> dict[str, object]:
    return {
        "seasons": sorted(seasons),
        "val_seasons": sorted(val_seasons),
        "perspective": perspective,
        "context_window": context_window,
        "max_seq_len": max_seq_len,
        "min_games": min_games,
        "target_mode": target_mode,
        "target_window": target_window,
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
            help="What to prepare: 'pretrain', 'finetune', 'hier-finetune', or 'all'",
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
    target_mode: Annotated[
        str,
        typer.Option(
            "--target-mode",
            help="Target mode for finetune: 'rates' (default) or 'counts' (legacy)",
        ),
    ] = "rates",
    target_window: Annotated[
        int,
        typer.Option(
            "--target-window",
            help="Number of games to average for target rate (rates mode only)",
        ),
    ] = 5,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            "-w",
            help="Max parallel workers for data building (default: CPU count)",
        ),
    ] = None,
    n_archetypes: Annotated[
        int,
        typer.Option(
            "--n-archetypes",
            help="Number of archetypes (hier-finetune mode only)",
        ),
    ] = 8,
    archetype_model_name: Annotated[
        str | None,
        typer.Option(
            "--archetype-model",
            help="Pre-fitted archetype model name (hier-finetune mode only)",
        ),
    ] = None,
    min_opportunities: Annotated[
        float,
        typer.Option(
            "--min-opportunities",
            help="Min career opportunities for profiles (hier-finetune mode only)",
        ),
    ] = 50.0,
    profile_year: Annotated[
        int | None,
        typer.Option(
            "--profile-year",
            help="'As of' year for profiles (hier-finetune mode; default: max val_seasons)",
        ),
    ] = None,
) -> None:
    """Pre-build tensorized sequences and save to disk.

    Saves prepared data so that pretrain/finetune/hier-finetune can skip
    expensive data building steps (parquet I/O, game sequence building,
    tensorization).

    Example:
        uv run fantasy-baseball-manager contextual prepare-data --mode all
        uv run fantasy-baseball-manager contextual prepare-data --mode hier-finetune --perspective pitcher
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
    do_hier = mode == "hier-finetune"

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
            ft_min_games = ft_context_window + target_window if target_mode == "rates" else ft_context_window + 5
            ft_config = FineTuneConfig(
                train_seasons=train_seasons,
                val_seasons=val_season_list,
                perspective=ft_perspective,
                context_window=ft_context_window,
                min_games=ft_min_games,
                target_mode=target_mode,
                target_window=target_window,
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
                target_mode=target_mode, target_window=target_window,
            )
            train_name = f"finetune_{ft_perspective}_train"
            val_name = f"finetune_{ft_perspective}_val"
            data_store.save_finetune_data(train_name, train_windows, ft_meta)
            data_store.save_finetune_data(val_name, val_windows, ft_meta)

            console.print(
                f"  Saved {len(train_windows)} train + {len(val_windows)} val windows"
            )

    if do_hier:
        from fantasy_baseball_manager.context import init_context
        from fantasy_baseball_manager.contextual.identity.archetypes import (
            fit_archetypes,
            load_archetype_model,
            save_archetype_model,
        )
        from fantasy_baseball_manager.contextual.identity.stat_profile import (
            PlayerStatProfileBuilder,
        )
        from fantasy_baseball_manager.contextual.training.config import (
            HierarchicalFineTuneConfig,
        )
        from fantasy_baseball_manager.contextual.training.hierarchical_dataset import (
            build_hierarchical_columnar,
        )
        from fantasy_baseball_manager.marcel.data_source import (
            create_batting_source,
            create_pitching_source,
        )

        resolved_profile_year = profile_year if profile_year is not None else max(val_season_list)
        init_context(year=resolved_profile_year)

        hier_context_window = context_window
        target_stats = BATTER_TARGET_STATS if perspective == "batter" else PITCHER_TARGET_STATS
        stat_input_dim = 19 if perspective == "batter" else 13

        console.print(f"[bold]Preparing hierarchical fine-tune data ({perspective})...[/bold]")

        # Build or load archetype model
        if archetype_model_name is not None:
            console.print(f"  Loading archetype model '{archetype_model_name}'...")
            arch_model = load_archetype_model(archetype_model_name)
        else:
            console.print("  Building stat profiles...")
            profile_builder = PlayerStatProfileBuilder()
            all_profiles = profile_builder.build_all_profiles(
                create_batting_source(), create_pitching_source(),
                resolved_profile_year, min_opportunities=min_opportunities,
            )
            profiles = [p for p in all_profiles if p.player_type == perspective]
            console.print(f"  {len(profiles)} {perspective} profiles")

            console.print("  Fitting archetypes...")
            arch_model, _labels = fit_archetypes(profiles, n_archetypes=n_archetypes)
            arch_name = f"{perspective}_archetypes"
            save_archetype_model(arch_model, arch_name)
            console.print(f"  Saved archetype model as '{arch_name}'")

        # Build profile lookup (need profiles even when loading archetype model)
        if archetype_model_name is not None:
            console.print("  Building stat profiles for lookup...")
            profile_builder = PlayerStatProfileBuilder()
            all_profiles = profile_builder.build_all_profiles(
                create_batting_source(), create_pitching_source(),
                resolved_profile_year, min_opportunities=min_opportunities,
            )
            profiles = [p for p in all_profiles if p.player_type == perspective]
            console.print(f"  {len(profiles)} {perspective} profiles")

        profile_lookup = {int(p.player_id): p for p in profiles}

        hier_min_games = hier_context_window + target_window if target_mode == "rates" else hier_context_window + 5
        hier_config = HierarchicalFineTuneConfig(
            train_seasons=train_seasons,
            val_seasons=val_season_list,
            perspective=perspective,
            context_window=hier_context_window,
            min_games=hier_min_games,
            target_mode=target_mode,
            target_window=target_window,
        )

        console.print("  Building training contexts...")
        train_contexts = build_player_contexts(
            builder, train_seasons, (perspective,), min_pitch_count=10,
            max_workers=workers,
        )
        console.print(f"  {len(train_contexts)} training player contexts")

        console.print("  Building hierarchical sliding windows (train) → columnar...")
        train_columnar = build_hierarchical_columnar(
            train_contexts, tensorizer, hier_config, target_stats,
            profile_lookup, arch_model, stat_input_dim,
        )
        del train_contexts
        n_train = int(train_columnar["seq_lengths"].shape[0])  # type: ignore[union-attr]
        console.print(f"  {n_train} training windows")

        console.print("  Building validation contexts...")
        val_contexts = build_player_contexts(
            builder, val_season_list, (perspective,), min_pitch_count=10,
            max_workers=workers,
        )
        console.print(f"  {len(val_contexts)} validation player contexts")

        console.print("  Building hierarchical sliding windows (val) → columnar...")
        val_columnar = build_hierarchical_columnar(
            val_contexts, tensorizer, hier_config, target_stats,
            profile_lookup, arch_model, stat_input_dim,
        )
        del val_contexts
        n_val = int(val_columnar["seq_lengths"].shape[0])  # type: ignore[union-attr]

        hier_meta = _build_finetune_meta(
            train_seasons, val_season_list, perspective, hier_context_window,
            max_seq_len, hier_config.min_games,
            target_mode=target_mode, target_window=target_window,
        )
        train_name = f"hier_finetune_{perspective}_train"
        val_name = f"hier_finetune_{perspective}_val"
        data_store.save_hierarchical_finetune_columnar(train_name, train_columnar, hier_meta)
        data_store.save_hierarchical_finetune_columnar(val_name, val_columnar, hier_meta)

        console.print(
            f"  Saved {n_train} train + {n_val} val hierarchical windows"
        )

    console.print("[bold green]Done![/bold green]")


def _print_classification_report(
    name: str,
    diag: object,
) -> None:
    """Print a classification report for one head."""
    from fantasy_baseball_manager.contextual.training.pretrain import ClassificationDiagnostics

    assert isinstance(diag, ClassificationDiagnostics)

    console.print(f"\n  [bold]{name}[/bold]")
    console.print(f"    Majority class:    {diag.majority_class} ({diag.majority_baseline:.1%})")
    console.print(f"    Model accuracy:    {diag.model_accuracy:.1%}")
    console.print(f"    Train accuracy:    {diag.train_accuracy:.1%}")
    gap = diag.train_accuracy - diag.model_accuracy
    gap_color = "red" if gap > 0.05 else "green"
    console.print(f"    Train-val gap:     [{gap_color}]{gap:+.1%}[/{gap_color}]")

    # Distribution sorted by frequency
    sorted_dist = sorted(diag.distribution.items(), key=lambda x: x[1], reverse=True)
    console.print("    Distribution:")
    for cls, count in sorted_dist:
        pct = count / sum(diag.distribution.values()) * 100
        console.print(f"      {cls:>25s}  {count:>6d}  ({pct:5.1f}%)")

    # Per-class report table
    console.print(f"    {'Class':>25s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'Support':>7s}")
    console.print(f"    {'─' * 25}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 7}")
    for cls, _count in sorted_dist:
        if cls in diag.report:
            r = diag.report[cls]
            console.print(
                f"    {cls:>25s}  {r['precision']:6.3f}  {r['recall']:6.3f}"
                f"  {r['f1-score']:6.3f}  {int(r['support']):7d}"
            )


def _print_diagnostics(diagnostics: object) -> None:
    """Print full pre-training diagnostics."""
    from fantasy_baseball_manager.contextual.training.pretrain import PreTrainDiagnostics

    assert isinstance(diagnostics, PreTrainDiagnostics)

    console.print("\n[bold]Validation Diagnostics[/bold]")
    _print_classification_report("Pitch Type", diagnostics.pitch_type)
    _print_classification_report("Pitch Result", diagnostics.pitch_result)


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
    d_model: Annotated[
        int,
        typer.Option(
            "--d-model",
            help="Transformer hidden dimension (default: 256)",
        ),
    ] = 256,
    n_layers: Annotated[
        int,
        typer.Option(
            "--n-layers",
            help="Number of transformer layers (default: 4)",
        ),
    ] = 4,
    n_heads: Annotated[
        int,
        typer.Option(
            "--n-heads",
            help="Number of attention heads (default: 8)",
        ),
    ] = 8,
    ff_dim: Annotated[
        int,
        typer.Option(
            "--ff-dim",
            help="Feed-forward hidden dimension (default: 1024)",
        ),
    ] = 1024,
) -> None:
    """Pre-train the contextual model using Masked Gamestate Modeling.

    Example:
        uv run fantasy-baseball-manager contextual pretrain --seasons 2015,2016,2017,2018,2019,2020,2021,2022 --val-seasons 2023

    For fast local iteration with a small model:
        uv run fantasy-baseball-manager contextual pretrain --d-model 64 --n-layers 2 --n-heads 2 --ff-dim 256 --max-seq-len 256 --epochs 5
    """
    import torch

    from fantasy_baseball_manager.contextual.data.vocab import (
        PA_EVENT_VOCAB,
        PITCH_RESULT_VOCAB,
        PITCH_TYPE_VOCAB,
    )
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.heads import MaskedGamestateHead
    from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
    from fantasy_baseball_manager.contextual.training.config import PreTrainingConfig
    from fantasy_baseball_manager.contextual.training.dataset import (
        MGMDataset,
        compute_feature_statistics,
    )
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
    console.print(f"  d_model:       {d_model}")
    console.print(f"  n_layers:      {n_layers}")
    console.print(f"  n_heads:       {n_heads}")
    console.print(f"  ff_dim:        {ff_dim}")
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

        model_config_for_tensorizer = ModelConfig(
            max_seq_len=max_seq_len, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, ff_dim=ff_dim,
        )
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

    model_config = ModelConfig(
        max_seq_len=max_seq_len, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, ff_dim=ff_dim,
    )

    train_dataset = MGMDataset(
        sequences=train_sequences,
        config=config,
        pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
        pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        pa_event_vocab_size=PA_EVENT_VOCAB.size,
    )
    val_dataset = MGMDataset(
        sequences=val_sequences,
        config=config,
        pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
        pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
        pa_event_vocab_size=PA_EVENT_VOCAB.size,
    )

    console.print(f"  {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    # Compute per-feature normalization statistics from training data
    console.print("  Computing feature statistics...")
    feature_mean, feature_std = compute_feature_statistics(train_sequences)

    # Build model
    console.print(f"  Device: {device}")

    head = MaskedGamestateHead(model_config)
    model = ContextualPerformanceModel(model_config, head)
    model.embedder.set_feature_statistics(feature_mean, feature_std)
    registry = _get_registry()
    model_store = registry.contextual_store

    trainer = MGMTrainer(model, model_config, config, model_store, device)

    console.print("\nStarting pre-training...")
    result = trainer.train(
        train_dataset, val_dataset, resume_from=resume_from,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
    )

    console.print("\n[bold]Training complete![/bold]")
    console.print(f"  Val loss:              {result['val_loss']:.4f}")
    console.print(f"  Val pitch type acc:    {result['val_pitch_type_accuracy']:.4f}")
    console.print(f"  Val pitch result acc:  {result['val_pitch_result_accuracy']:.4f}")

    if "diagnostics" in result:
        from fantasy_baseball_manager.contextual.training.pretrain import PreTrainDiagnostics

        diagnostics: PreTrainDiagnostics = result["diagnostics"]
        _print_diagnostics(diagnostics)


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
    target_mode: Annotated[
        str,
        typer.Option(
            "--target-mode",
            help="Target mode: 'rates' (default) or 'counts' (legacy)",
        ),
    ] = "rates",
    target_window: Annotated[
        int,
        typer.Option(
            "--target-window",
            help="Number of games to average for target rate (rates mode only)",
        ),
    ] = 5,
    prepared_data: Annotated[
        bool,
        typer.Option(
            "--prepared-data/--no-prepared-data",
            help="Load pre-built tensorized data if available",
        ),
    ] = True,
    d_model: Annotated[
        int,
        typer.Option(
            "--d-model",
            help="Transformer hidden dimension (must match pretrained model)",
        ),
    ] = 256,
    n_layers: Annotated[
        int,
        typer.Option(
            "--n-layers",
            help="Number of transformer layers (must match pretrained model)",
        ),
    ] = 4,
    n_heads: Annotated[
        int,
        typer.Option(
            "--n-heads",
            help="Number of attention heads (must match pretrained model)",
        ),
    ] = 8,
    ff_dim: Annotated[
        int,
        typer.Option(
            "--ff-dim",
            help="Feed-forward hidden dimension (must match pretrained model)",
        ),
    ] = 1024,
) -> None:
    """Fine-tune a pre-trained contextual model for per-game stat prediction.

    Example:
        uv run fantasy-baseball-manager contextual finetune --perspective pitcher --base-model pretrain_best

    For fast local iteration with a small model:
        uv run fantasy-baseball-manager contextual finetune --d-model 64 --n-layers 2 --n-heads 2 --ff-dim 256
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
    pretrain_state = torch.load(model_store._model_path(base_model), weights_only=True, map_location="cpu")
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

    ft_min_games = context_window + target_window if target_mode == "rates" else context_window + 5
    config = FineTuneConfig(
        train_seasons=train_seasons,
        val_seasons=val_season_list,
        perspective=perspective,
        context_window=context_window,
        min_games=ft_min_games,
        target_mode=target_mode,
        target_window=target_window,
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
    console.print(f"  Target mode:   {target_mode}")
    console.print(f"  Target window: {target_window}")
    console.print(f"  Context window: {context_window}")
    console.print(f"  Train seasons: {train_seasons}")
    console.print(f"  Val seasons:   {val_season_list}")
    console.print(f"  Epochs:        {epochs}")
    console.print(f"  Batch size:    {batch_size}")
    console.print(f"  Head LR:       {head_lr}")
    console.print(f"  Backbone LR:   {backbone_lr}")
    console.print(f"  Freeze backbone: {freeze_backbone}")
    console.print(f"  Max seq len:   {max_seq_len}")
    console.print(f"  d_model:       {d_model}")
    console.print(f"  n_layers:      {n_layers}")
    console.print(f"  n_heads:       {n_heads}")
    console.print(f"  ff_dim:        {ff_dim}")

    # Build model config and load pre-trained model
    model_config = ModelConfig(
        max_seq_len=max_seq_len, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, ff_dim=ff_dim,
    )

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
                target_mode=target_mode, target_window=target_window,
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


@contextual_app.command(name="build-identity")
def build_identity_cmd(
    perspective: Annotated[
        str,
        typer.Option(
            "--perspective",
            "-p",
            help="Player perspective: 'batter' or 'pitcher'",
        ),
    ] = "pitcher",
    n_archetypes: Annotated[
        int,
        typer.Option(
            "--n-archetypes",
            help="Number of player archetypes to fit",
        ),
    ] = 8,
    min_opportunities: Annotated[
        float,
        typer.Option(
            "--min-opportunities",
            help="Minimum career opportunities for profile inclusion",
        ),
    ] = 50.0,
    profile_year: Annotated[
        int,
        typer.Option(
            "--profile-year",
            help="'As of' year for profile building",
        ),
    ] = 2023,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            "-n",
            help="Name for saved archetype model (default: {perspective}_archetypes)",
        ),
    ] = None,
) -> None:
    """Build player stat profiles and fit an archetype clustering model.

    This creates the identity components needed by hier-finetune:
    stat profiles from season-level data, then KMeans archetype clustering
    on the profile feature vectors. The fitted archetype model is saved
    to disk and can be loaded by hier-finetune via --archetype-model.

    Example:
        uv run fantasy-baseball-manager contextual build-identity --perspective pitcher --n-archetypes 8
    """
    from fantasy_baseball_manager.contextual.identity.archetypes import (
        fit_archetypes,
        save_archetype_model,
    )
    from fantasy_baseball_manager.contextual.identity.stat_profile import (
        PlayerStatProfileBuilder,
    )
    from fantasy_baseball_manager.marcel.data_source import (
        create_batting_source,
        create_pitching_source,
    )

    model_name = name or f"{perspective}_archetypes"

    # Initialize ambient context (needed by data sources)
    from fantasy_baseball_manager.context import init_context

    init_context(year=profile_year)

    console.print("[bold]Building Player Identity (Profiles + Archetypes)[/bold]")
    console.print(f"  Perspective:       {perspective}")
    console.print(f"  Profile year:      {profile_year}")
    console.print(f"  N archetypes:      {n_archetypes}")
    console.print(f"  Min opportunities: {min_opportunities}")
    console.print(f"  Save as:           {model_name}")

    console.print("\nBuilding stat profiles...")
    builder = PlayerStatProfileBuilder()
    all_profiles = builder.build_all_profiles(
        create_batting_source(), create_pitching_source(),
        profile_year, min_opportunities=min_opportunities,
    )
    profiles = [p for p in all_profiles if p.player_type == perspective]
    console.print(f"  {len(profiles)} {perspective} profiles built")

    if not profiles:
        console.print("[red]No profiles found. Check perspective and data availability.[/red]")
        raise SystemExit(1)

    console.print("\nFitting archetypes...")
    arch_model, labels = fit_archetypes(profiles, n_archetypes=n_archetypes)

    # Print archetype distribution
    for arch_id in range(arch_model.n_archetypes):
        count = int((labels == arch_id).sum())
        examples = [p.name for p, lbl in zip(profiles, labels, strict=False) if lbl == arch_id][:3]
        example_str = ", ".join(examples)
        console.print(f"  Archetype {arch_id}: {count} players ({example_str})")

    save_path = save_archetype_model(arch_model, model_name)
    console.print(f"\n[bold green]Saved archetype model to {save_path}[/bold green]")
    console.print(f"  Use with: contextual hier-finetune --archetype-model {model_name}")


@contextual_app.command(name="hier-finetune")
def hier_finetune_cmd(
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
    identity_lr: Annotated[
        float,
        typer.Option(
            "--identity-lr",
            help="Learning rate for identity module",
        ),
    ] = 1e-3,
    level3_lr: Annotated[
        float,
        typer.Option(
            "--level3-lr",
            help="Learning rate for Level 3 attention",
        ),
    ] = 5e-4,
    head_lr: Annotated[
        float,
        typer.Option(
            "--head-lr",
            help="Learning rate for prediction head",
        ),
    ] = 1e-3,
    n_archetypes: Annotated[
        int,
        typer.Option(
            "--n-archetypes",
            help="Number of player archetypes",
        ),
    ] = 8,
    archetype_model_name: Annotated[
        str | None,
        typer.Option(
            "--archetype-model",
            help="Name of pre-fitted archetype model to load (fit fresh if omitted)",
        ),
    ] = None,
    min_opportunities: Annotated[
        float,
        typer.Option(
            "--min-opportunities",
            help="Minimum career opportunities for profile inclusion",
        ),
    ] = 50.0,
    profile_year: Annotated[
        int | None,
        typer.Option(
            "--profile-year",
            help="'As of' year for profile building (defaults to max val_seasons)",
        ),
    ] = None,
    max_seq_len: Annotated[
        int | None,
        typer.Option(
            "--max-seq-len",
            help="Maximum sequence length (auto-detected from pretrained model if omitted)",
        ),
    ] = None,
    target_mode: Annotated[
        str,
        typer.Option(
            "--target-mode",
            help="Target mode: 'rates' (default) or 'counts' (legacy)",
        ),
    ] = "rates",
    target_window: Annotated[
        int,
        typer.Option(
            "--target-window",
            help="Number of games to average for target rate (rates mode only)",
        ),
    ] = 5,
    d_model: Annotated[
        int,
        typer.Option(
            "--d-model",
            help="Transformer hidden dimension (must match pretrained model)",
        ),
    ] = 256,
    n_layers: Annotated[
        int,
        typer.Option(
            "--n-layers",
            help="Number of transformer layers (must match pretrained model)",
        ),
    ] = 4,
    n_heads: Annotated[
        int,
        typer.Option(
            "--n-heads",
            help="Number of attention heads (must match pretrained model)",
        ),
    ] = 8,
    ff_dim: Annotated[
        int,
        typer.Option(
            "--ff-dim",
            help="Feed-forward hidden dimension (must match pretrained model)",
        ),
    ] = 1024,
    prepared_data: Annotated[
        bool,
        typer.Option(
            "--prepared-data/--no-prepared-data",
            help="Load pre-built hierarchical windows if available",
        ),
    ] = True,
) -> None:
    """Fine-tune a hierarchical model with identity conditioning.

    Orchestrates three phases: build identity (profiles + archetypes),
    build hierarchical dataset, and train. The backbone is loaded from
    a pre-trained checkpoint and frozen during training.

    Pre-build data with: contextual prepare-data --mode hier-finetune

    Example:
        uv run fantasy-baseball-manager contextual hier-finetune --perspective pitcher --epochs 10
    """
    import torch

    from fantasy_baseball_manager.contextual.identity.archetypes import (
        fit_archetypes,
        load_archetype_model,
        save_archetype_model,
    )
    from fantasy_baseball_manager.contextual.identity.stat_profile import (
        PlayerStatProfileBuilder,
    )
    from fantasy_baseball_manager.contextual.model.config import ModelConfig
    from fantasy_baseball_manager.contextual.model.heads import PerformancePredictionHead
    from fantasy_baseball_manager.contextual.model.hierarchical import HierarchicalModel
    from fantasy_baseball_manager.contextual.model.hierarchical_config import (
        HierarchicalModelConfig,
    )
    from fantasy_baseball_manager.contextual.training.config import (
        BATTER_TARGET_STATS,
        DEFAULT_BATTER_CONTEXT_WINDOW,
        DEFAULT_PITCHER_CONTEXT_WINDOW,
        PITCHER_TARGET_STATS,
        HierarchicalFineTuneConfig,
    )
    from fantasy_baseball_manager.contextual.training.hierarchical_dataset import (
        HierarchicalFineTuneDataset,
        build_hierarchical_columnar,
    )
    from fantasy_baseball_manager.contextual.training.hierarchical_finetune import (
        HierarchicalFineTuneTrainer,
    )

    # Parse arguments
    train_seasons = tuple(int(s.strip()) for s in seasons.split(","))
    val_season_list = tuple(int(s.strip()) for s in val_seasons.split(","))

    if context_window is None:
        context_window = DEFAULT_BATTER_CONTEXT_WINDOW if perspective == "batter" else DEFAULT_PITCHER_CONTEXT_WINDOW

    target_stats = BATTER_TARGET_STATS if perspective == "batter" else PITCHER_TARGET_STATS
    n_targets = len(target_stats)

    stat_input_dim = 19 if perspective == "batter" else 13  # 6*3+1 or 4*3+1

    if profile_year is None:
        profile_year = max(val_season_list)

    # Auto-detect max_seq_len from pretrained model
    registry = _get_registry()
    model_store = registry.contextual_store
    pretrain_state = torch.load(model_store._model_path(base_model), weights_only=True, map_location="cpu")
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

    # Build configs
    model_config = ModelConfig(
        max_seq_len=max_seq_len, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, ff_dim=ff_dim,
    )

    hier_config = HierarchicalModelConfig(
        n_archetypes=n_archetypes,
        level3_d_model=d_model,
        batter_stat_input_dim=19,
        pitcher_stat_input_dim=13,
    )

    ft_min_games = context_window + target_window if target_mode == "rates" else context_window + 5
    ft_config = HierarchicalFineTuneConfig(
        train_seasons=train_seasons,
        val_seasons=val_season_list,
        perspective=perspective,
        context_window=context_window,
        min_games=ft_min_games,
        target_mode=target_mode,
        target_window=target_window,
        epochs=epochs,
        batch_size=batch_size,
        identity_learning_rate=identity_lr,
        level3_learning_rate=level3_lr,
        head_learning_rate=head_lr,
    )

    # Initialize ambient context (needed by data sources for profile building)
    from fantasy_baseball_manager.context import init_context

    init_context(year=profile_year)

    console.print("[bold]Hierarchical Fine-Tuning (Phase 2a)[/bold]")
    console.print(f"  Base model:      {base_model}")
    console.print(f"  Perspective:     {perspective}")
    console.print(f"  Target stats:    {target_stats}")
    console.print(f"  Target mode:     {target_mode}")
    console.print(f"  Target window:   {target_window}")
    console.print(f"  Context window:  {context_window}")
    console.print(f"  Train seasons:   {train_seasons}")
    console.print(f"  Val seasons:     {val_season_list}")
    console.print(f"  Epochs:          {epochs}")
    console.print(f"  Batch size:      {batch_size}")
    console.print(f"  Identity LR:     {identity_lr}")
    console.print(f"  Level 3 LR:      {level3_lr}")
    console.print(f"  Head LR:         {head_lr}")
    console.print(f"  Max seq len:     {max_seq_len}")
    console.print(f"  d_model:         {d_model}")
    console.print(f"  n_layers:        {n_layers}")
    console.print(f"  n_heads:         {n_heads}")
    console.print(f"  ff_dim:          {ff_dim}")
    console.print(f"  N archetypes:    {n_archetypes}")
    console.print(f"  Profile year:    {profile_year}")

    # Load pre-trained backbone
    console.print(f"\nLoading pre-trained model '{base_model}'...")
    backbone = model_store.load_model(base_model, model_config)

    # Swap head to PerformancePredictionHead (needed as backbone for HierarchicalModel)
    perf_head = PerformancePredictionHead(model_config, n_targets)
    backbone.swap_head(perf_head)
    console.print(f"  Swapped head to PerformancePredictionHead (n_targets={n_targets})")

    # Wrap backbone in HierarchicalModel
    model = HierarchicalModel(
        backbone=backbone,
        hier_config=hier_config,
        n_targets=n_targets,
        stat_input_dim=stat_input_dim,
    )

    # Try loading prepared data (columnar dict)
    train_dataset: HierarchicalFineTuneDataset | None = None
    val_dataset: HierarchicalFineTuneDataset | None = None

    if prepared_data:
        from fantasy_baseball_manager.contextual.data_store import PreparedDataStore

        data_store = PreparedDataStore()
        train_name = f"hier_finetune_{perspective}_train"
        val_name = f"hier_finetune_{perspective}_val"

        if data_store.exists(train_name) and data_store.exists(val_name):
            expected_meta = _build_finetune_meta(
                train_seasons, val_season_list, perspective, context_window,
                max_seq_len, ft_config.min_games,
                target_mode=target_mode, target_window=target_window,
            )
            stored_meta = data_store.load_meta(train_name)
            if stored_meta == expected_meta:
                console.print("\nLoading prepared hierarchical data...")
                train_data = data_store.load_hierarchical_finetune_data(train_name)
                val_data = data_store.load_hierarchical_finetune_data(val_name)
                n_train = int(train_data["seq_lengths"].shape[0])  # type: ignore[union-attr]
                n_val = int(val_data["seq_lengths"].shape[0])  # type: ignore[union-attr]
                console.print(
                    f"  Loaded prepared data ({n_train} train, "
                    f"{n_val} val windows)"
                )
                train_dataset = HierarchicalFineTuneDataset(train_data)
                val_dataset = HierarchicalFineTuneDataset(val_data)
            else:
                console.print("[red]Prepared data exists but parameters don't match:[/red]")
                _log_meta_mismatch(stored_meta, expected_meta)
                raise SystemExit(
                    "Re-run 'prepare-data --mode hier-finetune' with matching parameters, "
                    "or use --no-prepared-data to build inline."
                )

    if train_dataset is None or val_dataset is None:
        # Build or load identity
        if archetype_model_name is not None:
            console.print(f"\nLoading archetype model '{archetype_model_name}'...")
            arch_model = load_archetype_model(archetype_model_name)
            console.print(f"  Loaded ({arch_model.n_archetypes} archetypes)")
        else:
            console.print("\nBuilding stat profiles...")
            from fantasy_baseball_manager.marcel.data_source import (
                create_batting_source,
                create_pitching_source,
            )

            builder_profiles = PlayerStatProfileBuilder()
            all_profiles = builder_profiles.build_all_profiles(
                create_batting_source(), create_pitching_source(),
                profile_year, min_opportunities=min_opportunities,
            )
            profiles = [p for p in all_profiles if p.player_type == perspective]
            console.print(f"  {len(profiles)} {perspective} profiles built")

            console.print("  Fitting archetypes...")
            arch_model, _labels = fit_archetypes(profiles, n_archetypes=n_archetypes)
            arch_name = f"{perspective}_archetypes"
            save_archetype_model(arch_model, arch_name)
            console.print(f"  Saved archetype model as '{arch_name}' ({arch_model.n_archetypes} archetypes)")

        # Build profiles for player lookup (needed for both paths)
        if archetype_model_name is not None:
            console.print("  Building stat profiles for player lookup...")
            from fantasy_baseball_manager.marcel.data_source import (
                create_batting_source,
                create_pitching_source,
            )

            builder_profiles = PlayerStatProfileBuilder()
            all_profiles = builder_profiles.build_all_profiles(
                create_batting_source(), create_pitching_source(),
                profile_year, min_opportunities=min_opportunities,
            )
            profiles = [p for p in all_profiles if p.player_type == perspective]
            console.print(f"  {len(profiles)} {perspective} profiles built")

        profile_lookup = {int(p.player_id): p for p in profiles}

        # Build hierarchical windows
        console.print("\nBuilding training data...")
        from fantasy_baseball_manager.contextual.data.builder import GameSequenceBuilder
        from fantasy_baseball_manager.contextual.data.vocab import (
            BB_TYPE_VOCAB,
            HANDEDNESS_VOCAB,
            PA_EVENT_VOCAB,
            PITCH_RESULT_VOCAB,
            PITCH_TYPE_VOCAB,
        )
        from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
        from fantasy_baseball_manager.contextual.training.dataset import build_player_contexts
        from fantasy_baseball_manager.statcast.models import DEFAULT_DATA_DIR
        from fantasy_baseball_manager.statcast.store import StatcastStore

        store = StatcastStore(data_dir=DEFAULT_DATA_DIR)
        seq_builder = GameSequenceBuilder(store)

        tensorizer = Tensorizer(
            config=model_config,
            pitch_type_vocab=PITCH_TYPE_VOCAB,
            pitch_result_vocab=PITCH_RESULT_VOCAB,
            bb_type_vocab=BB_TYPE_VOCAB,
            handedness_vocab=HANDEDNESS_VOCAB,
            pa_event_vocab=PA_EVENT_VOCAB,
        )

        train_contexts = build_player_contexts(
            seq_builder, train_seasons, (perspective,), min_pitch_count=10,
        )
        console.print(f"  {len(train_contexts)} training player contexts")

        console.print("Building validation data...")
        val_contexts = build_player_contexts(
            seq_builder, val_season_list, (perspective,), min_pitch_count=10,
        )
        console.print(f"  {len(val_contexts)} validation player contexts")

        console.print("Building hierarchical sliding windows → columnar...")
        train_columnar = build_hierarchical_columnar(
            train_contexts, tensorizer, ft_config, target_stats,
            profile_lookup, arch_model, stat_input_dim,
        )
        del train_contexts
        val_columnar = build_hierarchical_columnar(
            val_contexts, tensorizer, ft_config, target_stats,
            profile_lookup, arch_model, stat_input_dim,
        )
        del val_contexts

        train_dataset = HierarchicalFineTuneDataset(train_columnar)
        val_dataset = HierarchicalFineTuneDataset(val_columnar)

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
    trainer = HierarchicalFineTuneTrainer(
        model, ft_config, model_store, target_stats, device,
    )

    console.print("\nStarting hierarchical fine-tuning...")
    result = trainer.train(train_dataset, val_dataset)

    console.print("\n[bold]Hierarchical fine-tuning complete![/bold]")
    console.print(f"  Val loss: {result['val_loss']:.4f}")
    for stat in target_stats:
        mse_key = f"val_{stat}_mse"
        mae_key = f"val_{stat}_mae"
        if mse_key in result:
            console.print(f"  {stat}: MSE={result[mse_key]:.4f}  MAE={result[mae_key]:.4f}")
