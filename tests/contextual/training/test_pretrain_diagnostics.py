"""Tests for pre-training validation diagnostics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from fantasy_baseball_manager.contextual.data.vocab import (
    BB_TYPE_VOCAB,
    HANDEDNESS_VOCAB,
    PA_EVENT_VOCAB,
    PITCH_RESULT_VOCAB,
    PITCH_TYPE_VOCAB,
    Vocabulary,
    _build_vocab,
)
from fantasy_baseball_manager.contextual.model.heads import MaskedGamestateHead
from fantasy_baseball_manager.contextual.model.model import ContextualPerformanceModel
from fantasy_baseball_manager.contextual.model.tensorizer import Tensorizer
from fantasy_baseball_manager.contextual.persistence import ContextualModelStore
from fantasy_baseball_manager.contextual.training.config import PreTrainingConfig
from fantasy_baseball_manager.contextual.training.dataset import MGMDataset
from fantasy_baseball_manager.contextual.training.pretrain import (
    MGMTrainer,
    PreTrainDiagnostics,
    compute_classification_diagnostics,
)
from tests.contextual.model.conftest import make_player_context

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from fantasy_baseball_manager.contextual.model.config import ModelConfig


class TestComputeClassificationDiagnostics:
    """Tests for the pure compute_classification_diagnostics function."""

    def _small_vocab(self) -> Vocabulary:
        return _build_vocab("test", ["A", "B", "C"])

    def test_majority_class(self) -> None:
        """Returns correct majority class and baseline accuracy."""
        vocab = self._small_vocab()
        # indices: 2=A, 3=B, 4=C
        # 6 A's, 3 B's, 1 C = majority is A at 60%
        targets = np.array([2, 2, 2, 2, 2, 2, 3, 3, 3, 4])
        preds = np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4])

        result = compute_classification_diagnostics(targets, preds, vocab, train_accuracy=0.9)

        assert result.majority_class == "A"
        assert result.majority_baseline == 0.6

    def test_excludes_pad_unk(self) -> None:
        """Indices 0 (PAD) and 1 (UNK) are filtered before computing metrics."""
        vocab = self._small_vocab()
        # Include PAD and UNK indices that should be excluded
        targets = np.array([0, 1, 2, 2, 2, 3, 3])
        preds = np.array([0, 1, 2, 2, 3, 3, 3])

        result = compute_classification_diagnostics(targets, preds, vocab, train_accuracy=0.8)

        # Only 5 valid samples (indices 2 and 3)
        total = sum(result.distribution.values())
        assert total == 5
        # PAD and UNK should not appear in distribution
        assert "<PAD>" not in result.distribution
        assert "<UNK>" not in result.distribution

    def test_report_structure(self) -> None:
        """Report dict contains per-class keys with precision/recall/f1/support."""
        vocab = self._small_vocab()
        targets = np.array([2, 2, 2, 3, 3, 4])
        preds = np.array([2, 2, 3, 3, 3, 4])

        result = compute_classification_diagnostics(targets, preds, vocab, train_accuracy=0.85)

        # Per-class keys should exist
        assert "A" in result.report
        assert "B" in result.report
        assert "C" in result.report

        # Each class should have standard classification metrics
        for cls in ["A", "B", "C"]:
            assert "precision" in result.report[cls]
            assert "recall" in result.report[cls]
            assert "f1-score" in result.report[cls]
            assert "support" in result.report[cls]

    def test_distribution_sums(self) -> None:
        """Distribution counts sum to total valid samples (after filtering)."""
        vocab = self._small_vocab()
        # 2 PAD, 1 UNK, 7 valid
        targets = np.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 4])
        preds = np.array([0, 0, 1, 2, 2, 3, 3, 3, 4, 2])

        result = compute_classification_diagnostics(targets, preds, vocab, train_accuracy=0.7)

        total = sum(result.distribution.values())
        assert total == 7

    def test_model_accuracy(self) -> None:
        """Model accuracy is correctly computed from filtered predictions."""
        vocab = self._small_vocab()
        targets = np.array([2, 2, 2, 3, 3])
        preds = np.array([2, 2, 3, 3, 3])

        result = compute_classification_diagnostics(targets, preds, vocab, train_accuracy=0.9)

        # 4 out of 5 correct
        assert result.model_accuracy == 0.8

    def test_train_accuracy_stored(self) -> None:
        """Train accuracy from the argument is stored in the result."""
        vocab = self._small_vocab()
        targets = np.array([2, 3])
        preds = np.array([2, 3])

        result = compute_classification_diagnostics(targets, preds, vocab, train_accuracy=0.95)

        assert result.train_accuracy == 0.95


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
    config: ModelConfig,
    train_config: PreTrainingConfig,
    n_samples: int = 4,
    n_games: int = 2,
    pitches_per_game: int = 15,
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


class TestTrainerDiagnostics:
    """Integration tests for diagnostics in the trainer."""

    def test_train_with_vocabs_returns_diagnostics(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """train() with vocabs returns PreTrainDiagnostics under 'diagnostics' key."""
        train_config = PreTrainingConfig(
            epochs=1, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        result = trainer.train(
            train_ds, val_ds,
            pitch_type_vocab=PITCH_TYPE_VOCAB,
            pitch_result_vocab=PITCH_RESULT_VOCAB,
        )

        assert "diagnostics" in result
        diag = result["diagnostics"]
        assert isinstance(diag, PreTrainDiagnostics)
        assert diag.pitch_type.majority_class != ""
        assert 0.0 <= diag.pitch_type.majority_baseline <= 1.0
        assert 0.0 <= diag.pitch_result.model_accuracy <= 1.0
        assert sum(diag.pitch_type.distribution.values()) > 0

    def test_train_without_vocabs_omits_diagnostics(self, small_config: ModelConfig, tmp_path: Path) -> None:
        """train() without vocabs has no 'diagnostics' key (backward compat)."""
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

        assert "diagnostics" not in result
        assert "val_loss" in result

    def test_per_epoch_log_includes_train_accuracy(
        self, small_config: ModelConfig, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Log lines contain train_pt_acc and train_pr_acc."""
        train_config = PreTrainingConfig(
            epochs=2, batch_size=2, learning_rate=1e-3, seed=42,
            min_warmup_steps=1, warmup_fraction=0.1,
            log_interval=1000,  # Suppress batch-level logs
        )
        model = ContextualPerformanceModel(small_config, MaskedGamestateHead(small_config))
        store = ContextualModelStore(model_dir=tmp_path)
        trainer = MGMTrainer(model, small_config, train_config, store)

        train_ds = _make_dataset(small_config, train_config, n_samples=4)
        val_ds = _make_dataset(small_config, train_config, n_samples=2)

        with caplog.at_level(logging.INFO, logger="fantasy_baseball_manager.contextual.training.pretrain"):
            trainer.train(train_ds, val_ds)

        epoch_lines = [r.message for r in caplog.records if "Epoch " in r.message]
        assert len(epoch_lines) >= 2
        for line in epoch_lines:
            assert "train_pt_acc=" in line
            assert "train_pr_acc=" in line
            assert "val_pt_acc=" in line
            assert "val_pr_acc=" in line
