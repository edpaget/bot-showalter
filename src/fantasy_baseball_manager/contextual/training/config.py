"""Configuration for contextual model pre-training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreTrainingConfig:
    """Configuration for Masked Gamestate Modeling pre-training."""

    # Data
    train_seasons: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
    val_seasons: tuple[int, ...] = (2023,)
    perspectives: tuple[str, ...] = ("batter", "pitcher")
    min_pitch_count: int = 10

    # Masking
    mask_ratio: float = 0.15
    mask_replace_ratio: float = 0.8  # 80% → zeroed
    mask_random_ratio: float = 0.1  # 10% → random
    # remaining 10% → keep original

    # Training
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    min_warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Loss
    pitch_type_loss_weight: float = 1.0
    pitch_result_loss_weight: float = 1.0

    # Checkpointing
    checkpoint_interval: int = 5
    log_interval: int = 100
    seed: int = 42
