"""Tests for PreTrainingConfig and FineTuneConfig."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.contextual.training.config import (
    DEFAULT_BATTER_CONTEXT_WINDOW,
    DEFAULT_PITCHER_CONTEXT_WINDOW,
    ContextualBlenderConfig,
    ContextualRateComputerConfig,
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
        assert config.target_mode == "rates"
        assert config.target_window == 5
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

    def test_min_games_must_exceed_context_window_counts_mode(self) -> None:
        config = FineTuneConfig(target_mode="counts")
        assert config.min_games >= config.context_window + 1

    def test_min_games_equals_context_window_raises_counts_mode(self) -> None:
        with pytest.raises(ValueError, match="min_games"):
            FineTuneConfig(target_mode="counts", context_window=10, min_games=10)

    def test_min_games_less_than_context_window_raises_counts_mode(self) -> None:
        with pytest.raises(ValueError, match="min_games"):
            FineTuneConfig(target_mode="counts", context_window=10, min_games=5)

    def test_rates_mode_min_games_validation(self) -> None:
        # context_window=10, target_window=5 → need min_games >= 15
        config = FineTuneConfig(
            target_mode="rates", context_window=10, target_window=5, min_games=15,
        )
        assert config.min_games == 15

    def test_rates_mode_min_games_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="min_games"):
            FineTuneConfig(
                target_mode="rates", context_window=10, target_window=5, min_games=14,
            )

    def test_invalid_target_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="target_mode"):
            FineTuneConfig(target_mode="invalid")

    def test_counts_mode_backward_compat(self) -> None:
        config = FineTuneConfig(
            target_mode="counts", context_window=10, min_games=11,
        )
        assert config.target_mode == "counts"
        assert config.min_games == 11


class TestContextualRateComputerConfig:
    def test_default_values(self) -> None:
        config = ContextualRateComputerConfig()
        assert config.batter_model_name == "finetune_batter_best"
        assert config.pitcher_model_name == "finetune_pitcher_best"
        assert config.batter_context_window == 30
        assert config.pitcher_context_window == 10
        assert config.batter_min_games == 30
        assert config.pitcher_min_games == 10

    def test_frozen(self) -> None:
        config = ContextualRateComputerConfig()
        with pytest.raises(AttributeError):
            config.batter_min_games = 5  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = ContextualRateComputerConfig(
            batter_model_name="custom_batter",
            pitcher_model_name="custom_pitcher",
            batter_context_window=20,
            pitcher_context_window=8,
            batter_min_games=25,
            pitcher_min_games=12,
        )
        assert config.batter_model_name == "custom_batter"
        assert config.pitcher_model_name == "custom_pitcher"
        assert config.batter_context_window == 20
        assert config.pitcher_context_window == 8
        assert config.batter_min_games == 25
        assert config.pitcher_min_games == 12

    def test_context_window_for_batter(self) -> None:
        config = ContextualRateComputerConfig()
        assert config.context_window_for("batter") == 30

    def test_context_window_for_pitcher(self) -> None:
        config = ContextualRateComputerConfig()
        assert config.context_window_for("pitcher") == 10

    def test_min_games_for_batter(self) -> None:
        config = ContextualRateComputerConfig()
        assert config.min_games_for("batter") == 30

    def test_min_games_for_pitcher(self) -> None:
        config = ContextualRateComputerConfig()
        assert config.min_games_for("pitcher") == 10


class TestContextualBlenderConfig:
    def test_default_values(self) -> None:
        config = ContextualBlenderConfig()
        assert config.batter_model_name == "finetune_batter_best"
        assert config.pitcher_model_name == "finetune_pitcher_best"
        assert config.batter_context_window == 30
        assert config.pitcher_context_window == 10
        assert config.batter_min_games == 30
        assert config.pitcher_min_games == 10
        assert config.contextual_weight == 0.3

    def test_frozen(self) -> None:
        config = ContextualBlenderConfig()
        with pytest.raises(AttributeError):
            config.contextual_weight = 0.5  # type: ignore[misc]

    def test_custom_values(self) -> None:
        config = ContextualBlenderConfig(
            batter_model_name="custom_batter",
            pitcher_model_name="custom_pitcher",
            batter_context_window=20,
            pitcher_context_window=8,
            batter_min_games=25,
            pitcher_min_games=12,
            contextual_weight=0.5,
        )
        assert config.batter_model_name == "custom_batter"
        assert config.pitcher_model_name == "custom_pitcher"
        assert config.batter_context_window == 20
        assert config.pitcher_context_window == 8
        assert config.batter_min_games == 25
        assert config.pitcher_min_games == 12
        assert config.contextual_weight == 0.5

    def test_context_window_for_batter(self) -> None:
        config = ContextualBlenderConfig()
        assert config.context_window_for("batter") == 30

    def test_context_window_for_pitcher(self) -> None:
        config = ContextualBlenderConfig()
        assert config.context_window_for("pitcher") == 10

    def test_min_games_for_batter(self) -> None:
        config = ContextualBlenderConfig()
        assert config.min_games_for("batter") == 30

    def test_min_games_for_pitcher(self) -> None:
        config = ContextualBlenderConfig()
        assert config.min_games_for("pitcher") == 10


class TestContextConstants:
    def test_default_batter_context_window(self) -> None:
        assert DEFAULT_BATTER_CONTEXT_WINDOW == 30

    def test_default_pitcher_context_window(self) -> None:
        assert DEFAULT_PITCHER_CONTEXT_WINDOW == 10
