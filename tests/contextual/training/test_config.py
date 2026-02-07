"""Tests for PreTrainingConfig and FineTuneConfig."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.contextual.training.config import (
    FineTuneConfig,
    PreTrainingConfig,
)


class TestPreTrainingConfig:
    def test_default_values(self) -> None:
        config = PreTrainingConfig()
        assert config.train_seasons == (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
        assert config.val_seasons == (2023,)
        assert config.perspectives == ("batter", "pitcher")
        assert config.min_pitch_count == 10
        assert config.mask_ratio == 0.15
        assert config.mask_replace_ratio == 0.8
        assert config.mask_random_ratio == 0.1
        assert config.epochs == 30
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.warmup_fraction == 0.05
        assert config.min_warmup_steps == 500
        assert config.max_grad_norm == 1.0
        assert config.pitch_type_loss_weight == 1.0
        assert config.pitch_result_loss_weight == 1.0
        assert config.checkpoint_interval == 5
        assert config.log_interval == 100
        assert config.seed == 42

    def test_frozen(self) -> None:
        config = PreTrainingConfig()
        try:
            config.seed = 123  # type: ignore[misc]
            raise AssertionError("Should not be able to set attribute on frozen dataclass")
        except AttributeError:
            pass

    def test_custom_values(self) -> None:
        config = PreTrainingConfig(
            train_seasons=(2020, 2021),
            val_seasons=(2022,),
            perspectives=("batter",),
            epochs=10,
            batch_size=64,
            learning_rate=5e-5,
        )
        assert config.train_seasons == (2020, 2021)
        assert config.val_seasons == (2022,)
        assert config.perspectives == ("batter",)
        assert config.epochs == 10
        assert config.batch_size == 64
        assert config.learning_rate == 5e-5

    def test_mask_ratios_sum_to_one_or_less(self) -> None:
        config = PreTrainingConfig()
        total = config.mask_replace_ratio + config.mask_random_ratio
        assert total <= 1.0
        keep_ratio = 1.0 - total
        assert abs(keep_ratio - 0.1) < 1e-9


class TestFineTuneConfig:
    def test_default_values(self) -> None:
        config = FineTuneConfig()
        assert config.train_seasons == (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
        assert config.val_seasons == (2023,)
        assert config.perspective == "pitcher"
        assert config.context_window == 10
        assert config.min_games == 15
        assert config.epochs == 30
        assert config.batch_size == 32
        assert config.head_learning_rate == 1e-3
        assert config.backbone_learning_rate == 1e-5
        assert config.freeze_backbone is False
        assert config.weight_decay == 0.01
        assert config.warmup_fraction == 0.05
        assert config.min_warmup_steps == 100
        assert config.max_grad_norm == 1.0
        assert config.patience == 5
        assert config.checkpoint_interval == 5
        assert config.log_interval == 100
        assert config.seed == 42

    def test_frozen(self) -> None:
        config = FineTuneConfig()
        with pytest.raises(AttributeError):
            config.seed = 123  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = FineTuneConfig(
            train_seasons=(2020, 2021),
            val_seasons=(2022,),
            perspective="batter",
            context_window=5,
            epochs=10,
            batch_size=64,
            head_learning_rate=5e-4,
            backbone_learning_rate=1e-6,
            freeze_backbone=True,
        )
        assert config.train_seasons == (2020, 2021)
        assert config.val_seasons == (2022,)
        assert config.perspective == "batter"
        assert config.context_window == 5
        assert config.epochs == 10
        assert config.batch_size == 64
        assert config.head_learning_rate == 5e-4
        assert config.backbone_learning_rate == 1e-6
        assert config.freeze_backbone is True

    def test_min_games_must_exceed_context_window(self) -> None:
        config = FineTuneConfig()
        assert config.min_games >= config.context_window + 1

    def test_min_games_equals_context_window_raises(self) -> None:
        with pytest.raises(ValueError, match="min_games"):
            FineTuneConfig(context_window=10, min_games=10)

    def test_min_games_less_than_context_window_raises(self) -> None:
        with pytest.raises(ValueError, match="min_games"):
            FineTuneConfig(context_window=10, min_games=5)
