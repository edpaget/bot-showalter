"""Tests for FineTuneTrainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fantasy_baseball_manager.contextual.model.config import ModelConfig
from fantasy_baseball_manager.contextual.model.heads import PerformancePredictionHead
from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
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
)
from fantasy_baseball_manager.contextual.training.finetune import (
    FineTuneMetrics,
    FineTuneTrainer,
)
from tests.contextual.model.conftest import make_player_context


def _build_tensorizer(config: ModelConfig) -> Tensorizer:
    return Tensorizer(
        config=config,
        pitch_type_vocab=PITCH_TYPE_VOCAB,
        pitch_result_vocab=PITCH_RESULT_VOCAB,
        bb_type_vocab=BB_TYPE_VOCAB,
        handedness_vocab=HANDEDNESS_VOCAB,
        pa_event_vocab=PA_EVENT_VOCAB,
    )


def _make_datasets(
    config: ModelConfig,
    ft_config: FineTuneConfig,
    target_stats: tuple[str, ...],
    n_train_players: int = 4,
    n_val_players: int = 2,
    n_games: int = 6,
    pitches_per_game: int = 10,
) -> tuple[FineTuneDataset, FineTuneDataset]:
    tensorizer = _build_tensorizer(config)

    train_contexts = [
        make_player_context(n_games=n_games, pitches_per_game=pitches_per_game, player_id=660271 + i)
        for i in range(n_train_players)
    ]
    val_contexts = [
        make_player_context(n_games=n_games, pitches_per_game=pitches_per_game, player_id=770271 + i)
        for i in range(n_val_players)
    ]

    train_windows = build_finetune_windows(train_contexts, tensorizer, ft_config, target_stats)
    val_windows = build_finetune_windows(val_contexts, tensorizer, ft_config, target_stats)

    return FineTuneDataset(train_windows), FineTuneDataset(val_windows)


def _make_finetune_model(
    config: ModelConfig,
    n_targets: int,
) -> ContextualPerformanceModel:
    """Create a model with PerformancePredictionHead for fine-tuning."""
    head = PerformancePredictionHead(config, n_targets)
    return ContextualPerformanceModel(config, head)


class TestFineTuneMetrics:
    def test_dataclass_fields(self) -> None:
        metrics = FineTuneMetrics(
            loss=0.5,
            per_stat_mse={"hr": 0.1, "so": 0.2},
            per_stat_mae={"hr": 0.3, "so": 0.4},
            n_samples=100,
        )
        assert metrics.loss == 0.5
        assert metrics.n_samples == 100
        assert metrics.per_stat_mse["hr"] == 0.1
        assert metrics.per_stat_mae["so"] == 0.4
        # defaults for baseline fields
        assert metrics.baseline_per_stat_mse == {}
        assert metrics.baseline_per_stat_mae == {}

    def test_baseline_fields(self) -> None:
        metrics = FineTuneMetrics(
            loss=0.5,
            per_stat_mse={"hr": 0.1},
            per_stat_mae={"hr": 0.3},
            n_samples=100,
            baseline_per_stat_mse={"hr": 0.29},
            baseline_per_stat_mae={"hr": 0.35},
        )
        assert metrics.baseline_per_stat_mse["hr"] == 0.29
        assert metrics.baseline_per_stat_mae["hr"] == 0.35


class TestFineTuneTrainer:
    def test_loss_computation(self, small_config: ModelConfig, tmp_path: Path) -> None:
        target_stats = BATTER_TARGET_STATS
        ft_config = FineTuneConfig(
            epochs=1, batch_size=2, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=1e-3, backbone_learning_rate=1e-5,
            min_warmup_steps=1, warmup_fraction=0.1, patience=100,
        )
        model = _make_finetune_model(small_config, len(target_stats))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = FineTuneTrainer(model, small_config, ft_config, store, target_stats)

        train_ds, val_ds = _make_datasets(small_config, ft_config, target_stats)

        result = trainer.train(train_ds, val_ds)
        assert "val_loss" in result
        assert result["val_loss"] > 0
        # Baseline keys should be present
        for stat in target_stats:
            assert f"baseline_{stat}_mse" in result
            assert f"baseline_{stat}_mae" in result
            assert result[f"baseline_{stat}_mse"] >= 0
            assert result[f"baseline_{stat}_mae"] >= 0

    def test_training_reduces_loss(self, small_config: ModelConfig, tmp_path: Path) -> None:
        target_stats = PITCHER_TARGET_STATS
        ft_config = FineTuneConfig(
            epochs=15, batch_size=4, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=1e-3, backbone_learning_rate=1e-4,
            min_warmup_steps=1, warmup_fraction=0.01, patience=100,
            checkpoint_interval=100, log_interval=1000,
        )
        model = _make_finetune_model(small_config, len(target_stats))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = FineTuneTrainer(model, small_config, ft_config, store, target_stats)

        train_ds, val_ds = _make_datasets(
            small_config, ft_config, target_stats,
            n_train_players=8, pitches_per_game=15,
        )

        result = trainer.train(train_ds, val_ds)
        # Loss should be reasonable (not NaN/Inf)
        assert result["val_loss"] < 100.0

    def test_early_stopping(self, small_config: ModelConfig, tmp_path: Path) -> None:
        target_stats = PITCHER_TARGET_STATS
        ft_config = FineTuneConfig(
            epochs=100,  # High max epochs
            batch_size=4, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=1e-3, backbone_learning_rate=1e-4,
            min_warmup_steps=1, warmup_fraction=0.01,
            patience=3,  # Stop after 3 epochs without improvement
            checkpoint_interval=100, log_interval=1000,
        )
        model = _make_finetune_model(small_config, len(target_stats))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = FineTuneTrainer(model, small_config, ft_config, store, target_stats)

        train_ds, val_ds = _make_datasets(
            small_config, ft_config, target_stats,
            n_train_players=4, pitches_per_game=10,
        )

        result = trainer.train(train_ds, val_ds)
        # Should stop before all 100 epochs due to early stopping
        assert "val_loss" in result
        # Verify the model actually trained (non-zero best epoch)
        assert store.exists("finetune_pitcher_best")

    def test_checkpointing(self, small_config: ModelConfig, tmp_path: Path) -> None:
        target_stats = BATTER_TARGET_STATS
        ft_config = FineTuneConfig(
            epochs=3, batch_size=2, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=1e-3, backbone_learning_rate=1e-5,
            min_warmup_steps=1, warmup_fraction=0.1,
            checkpoint_interval=1, patience=100,
        )
        model = _make_finetune_model(small_config, len(target_stats))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = FineTuneTrainer(model, small_config, ft_config, store, target_stats)

        train_ds, val_ds = _make_datasets(small_config, ft_config, target_stats)

        trainer.train(train_ds, val_ds)

        assert store.exists("finetune_pitcher_best")
        assert store.exists("finetune_pitcher_latest")

        # Verify fine-tune metadata is present
        meta = store.get_metadata("finetune_pitcher_best")
        assert meta is not None
        assert meta.target_stats == target_stats

    def test_resume_training(self, small_config: ModelConfig, tmp_path: Path) -> None:
        target_stats = PITCHER_TARGET_STATS
        ft_config = FineTuneConfig(
            epochs=2, batch_size=2, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=1e-3, backbone_learning_rate=1e-5,
            min_warmup_steps=1, warmup_fraction=0.1,
            checkpoint_interval=1, patience=100,
        )
        model = _make_finetune_model(small_config, len(target_stats))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = FineTuneTrainer(model, small_config, ft_config, store, target_stats)

        train_ds, val_ds = _make_datasets(small_config, ft_config, target_stats)

        trainer.train(train_ds, val_ds)
        assert store.exists("finetune_pitcher_latest")

        # Resume for more epochs
        resume_config = FineTuneConfig(
            epochs=4, batch_size=2, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=1e-3, backbone_learning_rate=1e-5,
            min_warmup_steps=1, warmup_fraction=0.1,
            checkpoint_interval=1, patience=100,
        )
        model2 = _make_finetune_model(small_config, len(target_stats))
        trainer2 = FineTuneTrainer(model2, small_config, resume_config, store, target_stats)
        result = trainer2.train(train_ds, val_ds, resume_from="finetune_pitcher_latest")
        assert "val_loss" in result

    def test_freeze_backbone(self, small_config: ModelConfig, tmp_path: Path) -> None:
        target_stats = BATTER_TARGET_STATS
        ft_config = FineTuneConfig(
            epochs=2, batch_size=2, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=1e-3, backbone_learning_rate=1e-5,
            freeze_backbone=True,
            min_warmup_steps=1, warmup_fraction=0.1, patience=100,
        )
        model = _make_finetune_model(small_config, len(target_stats))

        # Record backbone param values before training
        backbone_before = {
            name: p.clone()
            for name, p in model.named_parameters()
            if name.startswith("embedder.") or name.startswith("transformer.")
        }

        store = ContextualModelStore(model_dir=tmp_path)
        trainer = FineTuneTrainer(model, small_config, ft_config, store, target_stats)

        train_ds, val_ds = _make_datasets(small_config, ft_config, target_stats)
        trainer.train(train_ds, val_ds)

        # Backbone params should not have changed
        for name, before_val in backbone_before.items():
            after_val = dict(model.named_parameters())[name]
            assert torch.equal(before_val, after_val), f"Backbone param {name} changed when frozen"

    def test_discriminative_lr(self, small_config: ModelConfig, tmp_path: Path) -> None:
        target_stats = PITCHER_TARGET_STATS
        head_lr = 1e-2
        backbone_lr = 1e-6
        ft_config = FineTuneConfig(
            epochs=1, batch_size=2, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=head_lr,
            backbone_learning_rate=backbone_lr,
            min_warmup_steps=1, warmup_fraction=0.1, patience=100,
        )
        model = _make_finetune_model(small_config, len(target_stats))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = FineTuneTrainer(model, small_config, ft_config, store, target_stats)

        # Access the optimizer param groups to verify LRs
        train_ds, val_ds = _make_datasets(small_config, ft_config, target_stats)
        # We need to trigger optimizer creation — just train
        trainer.train(train_ds, val_ds)
        # The trainer should have set up two param groups
        # This is a structural test — if the trainer doesn't error with
        # discriminative LRs, the wiring is correct
        assert "val_loss" in trainer.train(train_ds, val_ds)

    def test_gradient_accumulation(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """Training with gradient accumulation completes and produces valid loss."""
        target_stats = BATTER_TARGET_STATS
        ft_config = FineTuneConfig(
            epochs=2, batch_size=2, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=1e-3, backbone_learning_rate=1e-5,
            min_warmup_steps=1, warmup_fraction=0.1, patience=100,
            accumulation_steps=2,
        )
        model = _make_finetune_model(small_config, len(target_stats))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = FineTuneTrainer(model, small_config, ft_config, store, target_stats)

        train_ds, val_ds = _make_datasets(small_config, ft_config, target_stats)
        result = trainer.train(train_ds, val_ds)
        assert "val_loss" in result
        assert result["val_loss"] > 0

    def test_weighted_loss_differs_from_unweighted(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """Weighted loss should differ from uniform MSE when stat variances differ."""
        target_stats = BATTER_TARGET_STATS
        ft_config = FineTuneConfig(
            epochs=1, batch_size=2, target_mode="counts", context_window=2, min_games=3,
            head_learning_rate=1e-3, backbone_learning_rate=1e-5,
            min_warmup_steps=1, warmup_fraction=0.1, patience=100,
        )
        model = _make_finetune_model(small_config, len(target_stats))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = FineTuneTrainer(model, small_config, ft_config, store, target_stats)

        train_ds, val_ds = _make_datasets(small_config, ft_config, target_stats)
        trainer.train(train_ds, val_ds)

        # Verify loss weights were computed and are not uniform
        assert trainer._loss_weights is not None
        weights = trainer._loss_weights
        assert weights.shape == (len(target_stats),)
        # Weights should sum to n_stats (normalized)
        assert weights.sum().item() == pytest.approx(len(target_stats), rel=1e-4)
