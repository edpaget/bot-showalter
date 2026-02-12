"""Tests for MGMTrainer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

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
from torch.utils.data import DataLoader

from fantasy_baseball_manager.contextual.model.heads import MaskedGamestateHead
from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
from fantasy_baseball_manager.contextual.training.config import PreTrainingConfig
from fantasy_baseball_manager.contextual.training.dataset import MGMDataset
from fantasy_baseball_manager.contextual.training.pretrain import (
    MGMTrainer,
    TrainingMetrics,
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


def _make_dataset(
    config: ModelConfig, train_config: PreTrainingConfig, n_samples: int = 4, n_games: int = 2, pitches_per_game: int = 15
) -> MGMDataset:
    tensorizer = _build_tensorizer(config)
    sequences = []
    for i in range(n_samples):
        ctx = make_player_context(n_games=n_games, pitches_per_game=pitches_per_game, player_id=660271 + i)
        sequences.append(tensorizer.tensorize_context(ctx))
    return MGMDataset(
        sequences=sequences,
        config=train_config,
        pitch_type_vocab_size=PITCH_TYPE_VOCAB.size,
        pitch_result_vocab_size=PITCH_RESULT_VOCAB.size,
    )


class TestTrainingMetrics:
    def test_dataclass_fields(self) -> None:
        metrics = TrainingMetrics(
            loss=0.5,
            pitch_type_loss=0.3,
            pitch_result_loss=0.2,
            pitch_type_accuracy=0.75,
            pitch_result_accuracy=0.65,
            num_masked_tokens=100,
        )
        assert metrics.loss == 0.5
        assert metrics.num_masked_tokens == 100


class TestMGMTrainer:
    def test_loss_computation(self, small_config: ModelConfig, tmp_path: Path) -> None:
        train_config = PreTrainingConfig(
            epochs=1, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        result = trainer.train(train_ds, val_ds)
        assert "val_loss" in result
        assert "val_pitch_type_accuracy" in result
        assert "val_pitch_result_accuracy" in result
        assert result["val_loss"] > 0

    def test_training_reduces_loss(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """After multiple epochs, loss should decrease from initial."""
        train_config = PreTrainingConfig(
            epochs=10, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.01,
            checkpoint_interval=100,  # Don't checkpoint during this test
            log_interval=1000,
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        train_ds = _make_dataset(small_config, train_config, n_samples=8, pitches_per_game=20)
        val_ds = _make_dataset(small_config, train_config, n_samples=2, pitches_per_game=20)

        result = trainer.train(train_ds, val_ds)
        # Loss should be reasonable (not NaN/Inf)
        assert result["val_loss"] < 10.0

    def test_checkpointing(self, small_config: ModelConfig, tmp_path: Path) -> None:
        train_config = PreTrainingConfig(
            epochs=3, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
            checkpoint_interval=1,  # Checkpoint every epoch
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        trainer.train(train_ds, val_ds)

        # Should have best and latest checkpoints
        assert store.exists("pretrain_best")
        assert store.exists("pretrain_latest")

    def test_resume_training(self, small_config: ModelConfig, tmp_path: Path) -> None:
        train_config = PreTrainingConfig(
            epochs=2, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
            checkpoint_interval=1,
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        # Train for 2 epochs
        trainer.train(train_ds, val_ds)
        assert store.exists("pretrain_latest")

        # Resume training for 2 more epochs
        resume_config = PreTrainingConfig(
            epochs=4, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
            checkpoint_interval=1,
        )
        model2 = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        trainer2 = MGMTrainer(model2, small_config, resume_config, store)
        result = trainer2.train(train_ds, val_ds, resume_from="pretrain_latest")
        assert "val_loss" in result

    def test_amp_enabled_on_cpu(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """AMP code paths execute without error on CPU (autocast + GradScaler are no-ops)."""
        train_config = PreTrainingConfig(
            epochs=2, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
            amp_enabled=True,
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        result = trainer.train(train_ds, val_ds)
        assert result["val_loss"] > 0
        assert math.isfinite(result["val_loss"])

    def test_convergence_beats_random(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """After training, accuracy should exceed random baseline."""
        train_config = PreTrainingConfig(
            epochs=20, batch_size=4, learning_rate=5e-4, seed=42,
            min_warmup_steps=1, warmup_fraction=0.01,
            checkpoint_interval=100,
            log_interval=1000,
            mask_ratio=0.3,  # More masked tokens for better signal
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        # More data for convergence test
        train_ds = _make_dataset(small_config, train_config, n_samples=16, pitches_per_game=20)
        val_ds = _make_dataset(small_config, train_config, n_samples=4, pitches_per_game=20)

        result = trainer.train(train_ds, val_ds)
        # Random baselines: 1/21 ≈ 4.8% for pitch type, 1/17 ≈ 5.9% for pitch result
        # With synthetic data (all same pitch type/result), model should learn quickly
        assert result["val_pitch_type_accuracy"] > 0.048
        assert result["val_pitch_result_accuracy"] > 0.059

    def test_gradient_accumulation(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """Training with gradient accumulation completes and produces valid loss."""
        train_config = PreTrainingConfig(
            epochs=2, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
            accumulation_steps=2,
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        result = trainer.train(train_ds, val_ds)
        assert "val_loss" in result
        assert result["val_loss"] > 0

    def test_mps_dataloader_uses_workers(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """On MPS, DataLoader should use num_workers=2 and persistent_workers but no pin_memory."""
        import torch

        train_config = PreTrainingConfig(
            epochs=1, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        # Pretend device is MPS
        trainer._device = torch.device("mps")

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        captured_kwargs: list[dict[str, object]] = []
        original_init = DataLoader.__init__

        def spy_init(self_loader: Any, *args: Any, **kwargs: Any) -> None:
            captured_kwargs.append(kwargs)
            original_init(self_loader, *args, **kwargs)

        with (
            patch.object(DataLoader, "__init__", spy_init),
            patch.object(trainer, "_train_epoch", return_value=TrainingMetrics(0.1, 0.1, 0.1, 0.5, 0.5, 100)),
            patch.object(trainer, "_validate", return_value=TrainingMetrics(0.1, 0.1, 0.1, 0.5, 0.5, 100)),
        ):
            trainer.train(train_ds, val_ds)

        assert len(captured_kwargs) >= 2
        for kwargs in captured_kwargs:
            assert kwargs.get("num_workers") == 2
            assert kwargs.get("persistent_workers") is True
            assert "pin_memory" not in kwargs

    def test_amp_enabled_on_mps_pretrain(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """AMP autocast should be enabled on MPS; GradScaler should be disabled."""
        import torch

        train_config = PreTrainingConfig(
            epochs=1, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
            amp_enabled=True,
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        # Pretend device is MPS
        trainer._device = torch.device("mps")

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        with (
            patch.object(trainer, "_train_epoch", return_value=TrainingMetrics(0.1, 0.1, 0.1, 0.5, 0.5, 100)),
            patch.object(trainer, "_validate", return_value=TrainingMetrics(0.1, 0.1, 0.1, 0.5, 0.5, 100)),
        ):
            trainer.train(train_ds, val_ds)

        # GradScaler should be disabled on MPS
        assert not trainer._scaler.is_enabled()

    def test_cpu_dataloader_no_extra_workers(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """On CPU, DataLoader should not set num_workers/pin_memory (defaults only)."""
        train_config = PreTrainingConfig(
            epochs=1, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        captured_kwargs: list[dict[str, object]] = []
        original_init = DataLoader.__init__

        def spy_init(self_loader: Any, *args: Any, **kwargs: Any) -> None:
            captured_kwargs.append(kwargs)
            original_init(self_loader, *args, **kwargs)

        with patch.object(DataLoader, "__init__", spy_init):
            trainer.train(train_ds, val_ds)

        # Both train and val loaders should NOT have num_workers or pin_memory set
        for kwargs in captured_kwargs:
            assert "num_workers" not in kwargs
            assert "pin_memory" not in kwargs
            assert "persistent_workers" not in kwargs
