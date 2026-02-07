"""Tests for ModelConfig."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.contextual.model.config import ModelConfig


class TestModelConfig:
    """Tests for the ModelConfig frozen dataclass."""

    def test_default_values(self) -> None:
        config = ModelConfig()
        assert config.d_model == 256
        assert config.n_layers == 4
        assert config.n_heads == 8
        assert config.ff_dim == 1024
        assert config.dropout == 0.1
        assert config.max_seq_len == 2048

    def test_embedding_dims(self) -> None:
        config = ModelConfig()
        assert config.pitch_type_embed_dim == 16
        assert config.pitch_result_embed_dim == 12
        assert config.bb_type_embed_dim == 8
        assert config.stand_embed_dim == 4
        assert config.p_throws_embed_dim == 4
        assert config.pa_event_embed_dim == 16

    def test_vocab_sizes(self) -> None:
        config = ModelConfig()
        assert config.pitch_type_vocab_size == 21
        assert config.pitch_result_vocab_size == 17
        assert config.bb_type_vocab_size == 6
        assert config.handedness_vocab_size == 4
        assert config.pa_event_vocab_size == 27

    def test_numeric_features(self) -> None:
        config = ModelConfig()
        assert config.n_numeric_features == 23

    def test_target_counts(self) -> None:
        config = ModelConfig()
        assert config.n_batter_targets == 7
        assert config.n_pitcher_targets == 5

    def test_frozen_immutability(self) -> None:
        config = ModelConfig()
        with pytest.raises(AttributeError):
            config.d_model = 128  # type: ignore[misc]

    def test_head_dim_divisibility(self) -> None:
        config = ModelConfig()
        assert config.d_model % config.n_heads == 0

    def test_custom_values(self) -> None:
        config = ModelConfig(d_model=32, n_layers=1, n_heads=2, ff_dim=64)
        assert config.d_model == 32
        assert config.n_layers == 1
        assert config.n_heads == 2
        assert config.ff_dim == 64
