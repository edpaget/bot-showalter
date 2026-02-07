"""Tests for MaskedGamestateHead and PerformancePredictionHead."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fantasy_baseball_manager.contextual.model.heads import (
    MaskedGamestateHead,
    PerformancePredictionHead,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.contextual.model.config import ModelConfig


class TestMaskedGamestateHead:
    """Tests for the masked gamestate pre-training head."""

    def test_output_shapes(self, small_config: ModelConfig) -> None:
        head = MaskedGamestateHead(small_config)
        hidden = torch.randn(2, 10, small_config.d_model)
        pitch_type_logits, pitch_result_logits = head(hidden)
        assert pitch_type_logits.shape == (2, 10, small_config.pitch_type_vocab_size)
        assert pitch_result_logits.shape == (2, 10, small_config.pitch_result_vocab_size)

    def test_single_position_input(self, small_config: ModelConfig) -> None:
        head = MaskedGamestateHead(small_config)
        hidden = torch.randn(1, 1, small_config.d_model)
        pitch_type_logits, pitch_result_logits = head(hidden)
        assert pitch_type_logits.shape == (1, 1, small_config.pitch_type_vocab_size)
        assert pitch_result_logits.shape == (1, 1, small_config.pitch_result_vocab_size)

    def test_gradient_flow(self, small_config: ModelConfig) -> None:
        head = MaskedGamestateHead(small_config)
        hidden = torch.randn(1, 5, small_config.d_model, requires_grad=True)
        pitch_type_logits, pitch_result_logits = head(hidden)
        loss = pitch_type_logits.sum() + pitch_result_logits.sum()
        loss.backward()
        assert hidden.grad is not None
        assert hidden.grad.abs().sum() > 0


class TestPerformancePredictionHead:
    """Tests for the performance prediction fine-tuning head."""

    def test_batter_output_shape(self, small_config: ModelConfig) -> None:
        head = PerformancePredictionHead(small_config, n_targets=small_config.n_batter_targets)
        player_emb = torch.randn(2, 3, small_config.d_model)
        out = head(player_emb)
        assert out.shape == (2, 3, small_config.n_batter_targets)

    def test_pitcher_output_shape(self, small_config: ModelConfig) -> None:
        head = PerformancePredictionHead(small_config, n_targets=small_config.n_pitcher_targets)
        player_emb = torch.randn(2, 3, small_config.d_model)
        out = head(player_emb)
        assert out.shape == (2, 3, small_config.n_pitcher_targets)

    def test_single_position_input(self, small_config: ModelConfig) -> None:
        head = PerformancePredictionHead(small_config, n_targets=7)
        player_emb = torch.randn(1, 1, small_config.d_model)
        out = head(player_emb)
        assert out.shape == (1, 1, 7)

    def test_gradient_flow(self, small_config: ModelConfig) -> None:
        head = PerformancePredictionHead(small_config, n_targets=7)
        player_emb = torch.randn(1, 2, small_config.d_model, requires_grad=True)
        out = head(player_emb)
        loss = out.sum()
        loss.backward()
        assert player_emb.grad is not None
        assert player_emb.grad.abs().sum() > 0
