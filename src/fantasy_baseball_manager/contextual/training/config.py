"""Configuration for contextual model pre-training and fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

BATTER_TARGET_STATS: tuple[str, ...] = ("hr", "so", "bb", "h", "2b", "3b")
PITCHER_TARGET_STATS: tuple[str, ...] = ("so", "h", "bb", "hr")

DEFAULT_BATTER_CONTEXT_WINDOW: int = 30
DEFAULT_PITCHER_CONTEXT_WINDOW: int = 10


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
    accumulation_steps: int = 1

    # Automatic Mixed Precision
    amp_enabled: bool = False

    # Loss
    pitch_type_loss_weight: float = 1.0
    pitch_result_loss_weight: float = 1.0

    # Checkpointing
    checkpoint_interval: int = 5
    log_interval: int = 100
    seed: int = 42


@dataclass(frozen=True, slots=True)
class FineTuneConfig:
    """Configuration for fine-tuning on per-game stat prediction."""

    # Data
    train_seasons: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
    val_seasons: tuple[int, ...] = (2023,)
    perspective: str = "pitcher"
    context_window: int = 10
    min_games: int = 15
    target_mode: str = "rates"  # "counts" (legacy) | "rates"
    target_window: int = 5  # number of games to average for target rate

    # Training
    epochs: int = 30
    batch_size: int = 32
    head_learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-5
    freeze_backbone: bool = False
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    min_warmup_steps: int = 100
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1

    # Early stopping
    patience: int = 5

    # Checkpointing
    checkpoint_interval: int = 5
    log_interval: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        if self.target_mode not in ("counts", "rates"):
            msg = f"target_mode must be 'counts' or 'rates', got '{self.target_mode}'"
            raise ValueError(msg)
        if self.target_mode == "rates":
            required = self.context_window + self.target_window
            if self.min_games < required:
                msg = (
                    f"min_games ({self.min_games}) must be >= "
                    f"context_window + target_window ({required}) in rates mode"
                )
                raise ValueError(msg)
        else:
            if self.min_games < self.context_window + 1:
                msg = (
                    f"min_games ({self.min_games}) must be >= "
                    f"context_window + 1 ({self.context_window + 1})"
                )
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ContextualRateComputerConfig:
    """Configuration for contextual transformer rate computer in the pipeline."""

    batter_model_name: str = "finetune_batter_best"
    pitcher_model_name: str = "finetune_pitcher_best"
    batter_context_window: int = DEFAULT_BATTER_CONTEXT_WINDOW
    pitcher_context_window: int = DEFAULT_PITCHER_CONTEXT_WINDOW
    batter_min_games: int = DEFAULT_BATTER_CONTEXT_WINDOW
    pitcher_min_games: int = DEFAULT_PITCHER_CONTEXT_WINDOW
    rate_mode: bool = True

    def context_window_for(self, perspective: str) -> int:
        """Return context window size for the given perspective."""
        return self.batter_context_window if perspective == "batter" else self.pitcher_context_window

    def min_games_for(self, perspective: str) -> int:
        """Return minimum games required for the given perspective."""
        return self.batter_min_games if perspective == "batter" else self.pitcher_min_games


@dataclass(frozen=True, slots=True)
class ContextualBlenderConfig:
    """Configuration for contextual transformer blender in the pipeline."""

    batter_model_name: str = "finetune_batter_best"
    pitcher_model_name: str = "finetune_pitcher_best"
    batter_context_window: int = DEFAULT_BATTER_CONTEXT_WINDOW
    pitcher_context_window: int = DEFAULT_PITCHER_CONTEXT_WINDOW
    batter_min_games: int = DEFAULT_BATTER_CONTEXT_WINDOW
    pitcher_min_games: int = DEFAULT_PITCHER_CONTEXT_WINDOW
    contextual_weight: float = 0.3

    def context_window_for(self, perspective: str) -> int:
        """Return context window size for the given perspective."""
        return self.batter_context_window if perspective == "batter" else self.pitcher_context_window

    def min_games_for(self, perspective: str) -> int:
        """Return minimum games required for the given perspective."""
        return self.batter_min_games if perspective == "batter" else self.pitcher_min_games


@dataclass(frozen=True, slots=True)
class HierarchicalFineTuneConfig:
    """Configuration for hierarchical model fine-tuning (Phase 2a).

    Same data/training structure as FineTuneConfig, with separate LR fields
    for identity, level3, and head parameter groups.
    """

    # Data
    train_seasons: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
    val_seasons: tuple[int, ...] = (2023,)
    perspective: str = "pitcher"
    context_window: int = 10
    min_games: int = 15
    target_mode: str = "rates"
    target_window: int = 5

    # Training — per-group learning rates
    epochs: int = 30
    batch_size: int = 32
    identity_learning_rate: float = 1e-3
    level3_learning_rate: float = 5e-4
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    min_warmup_steps: int = 100
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1

    # Identity conditioning
    stat_feature_dropout: float = 0.0  # 0 for Phase 2a, 0.2 for Phase 2c

    # Early stopping
    patience: int = 5

    # Checkpointing
    checkpoint_interval: int = 5
    log_interval: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        if self.target_mode not in ("counts", "rates"):
            msg = f"target_mode must be 'counts' or 'rates', got '{self.target_mode}'"
            raise ValueError(msg)
        if self.target_mode == "rates":
            required = self.context_window + self.target_window
            if self.min_games < required:
                msg = (
                    f"min_games ({self.min_games}) must be >= "
                    f"context_window + target_window ({required}) in rates mode"
                )
                raise ValueError(msg)
        else:
            if self.min_games < self.context_window + 1:
                msg = (
                    f"min_games ({self.min_games}) must be >= "
                    f"context_window + 1 ({self.context_window + 1})"
                )
                raise ValueError(msg)
