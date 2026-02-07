"""Shared fixtures for contextual training tests."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.contextual.model.config import ModelConfig

# Re-export helpers from model conftest so training tests can use them directly.
from tests.contextual.model.conftest import (
    make_game_sequence,
    make_pitch,
    make_player_context,
)

__all__ = ["make_game_sequence", "make_pitch", "make_player_context"]


@pytest.fixture
def small_config() -> ModelConfig:
    """A small ModelConfig for fast tests."""
    return ModelConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        ff_dim=64,
        dropout=0.0,
        max_seq_len=128,
        pitch_type_embed_dim=8,
        pitch_result_embed_dim=6,
        bb_type_embed_dim=4,
        stand_embed_dim=4,
        p_throws_embed_dim=4,
        pa_event_embed_dim=8,
    )
