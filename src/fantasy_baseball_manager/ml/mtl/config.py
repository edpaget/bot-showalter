"""Configuration dataclasses for Multi-Task Learning models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MTLArchitectureConfig:
    """Neural network architecture configuration.

    Defines the shared trunk and per-stat output head architecture.
    Default architecture:
        - Shared trunk: Input → 64 → 32 (NO batch norm, minimal dropout)
        - Per-stat heads: 32 → 16 → 1

    Note: Batch normalization disabled - it can cause train/test mismatch
    with small datasets. Minimal dropout to preserve variance.
    """

    shared_layers: tuple[int, ...] = (64, 32)
    head_hidden_size: int = 16
    dropout_rates: tuple[float, ...] = (0.05, 0.0)
    use_batch_norm: bool = False


@dataclass(frozen=True)
class MTLTrainingConfig:
    """Training configuration for MTL models.

    Controls training hyperparameters, early stopping, and data filtering.

    Note: Very low weight decay to avoid over-regularization.
    """

    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 25
    val_fraction: float = 0.15
    min_samples: int = 50
    batter_min_pa: int = 100
    pitcher_min_pa: int = 100


@dataclass(frozen=True)
class MTLRateComputerConfig:
    """Configuration for MTL as a standalone rate computer.

    Used when MTL model replaces Marcel entirely for rate computation.
    """

    model_name: str = "default"
    min_pa: int = 100


@dataclass(frozen=True)
class MTLBlenderConfig:
    """Configuration for blending MTL predictions with Marcel rates.

    Used in ensemble mode: blended_rate = (1 - mtl_weight) * marcel + mtl_weight * mtl
    """

    model_name: str = "default"
    mtl_weight: float = 0.3
    min_pa: int = 100
