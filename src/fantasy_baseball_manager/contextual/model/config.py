"""Model configuration for the contextual performance transformer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for the contextual performance transformer model.

    All vocab sizes must match the corresponding Vocabulary.size values
    from contextual.data.vocab.
    """

    # Transformer core
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 2048

    # Categorical embedding dims
    pitch_type_embed_dim: int = 16
    pitch_result_embed_dim: int = 12
    bb_type_embed_dim: int = 8
    stand_embed_dim: int = 4
    p_throws_embed_dim: int = 4
    pa_event_embed_dim: int = 16

    # Numeric features (23 total from PitchEvent)
    n_numeric_features: int = 23

    # Vocab sizes (must match actual Vocabulary.size values)
    pitch_type_vocab_size: int = 21
    pitch_result_vocab_size: int = 17
    bb_type_vocab_size: int = 6
    handedness_vocab_size: int = 4
    pa_event_vocab_size: int = 27

    # Performance head targets (must match BATTER/PITCHER_TARGET_STATS in training.config)
    n_batter_targets: int = 6  # HR, SO, BB, H, 2B, 3B
    n_pitcher_targets: int = 4  # SO, H, BB, HR
