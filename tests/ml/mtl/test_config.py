"""Tests for MTL configuration dataclasses."""

from fantasy_baseball_manager.ml.mtl.config import (
    MTLArchitectureConfig,
    MTLBlenderConfig,
    MTLRateComputerConfig,
)


class TestMTLArchitectureConfig:
    def test_custom_values(self) -> None:
        config = MTLArchitectureConfig(
            shared_layers=(128, 64, 32),
            head_hidden_size=32,
            dropout_rates=(0.5, 0.4, 0.3),
            use_batch_norm=False,
        )
        assert config.shared_layers == (128, 64, 32)
        assert config.head_hidden_size == 32
        assert config.use_batch_norm is False

    def test_frozen(self) -> None:
        config = MTLArchitectureConfig()
        try:
            config.head_hidden_size = 32  # type: ignore
            raise AssertionError("Should raise FrozenInstanceError")
        except AttributeError:
            pass


class TestMTLRateComputerConfig:
    def test_default_values(self) -> None:
        config = MTLRateComputerConfig()
        assert config.model_name == "default"
        assert config.min_pa == 100

    def test_custom_values(self) -> None:
        config = MTLRateComputerConfig(model_name="custom", min_pa=200)
        assert config.model_name == "custom"
        assert config.min_pa == 200


class TestMTLBlenderConfig:
    def test_default_values(self) -> None:
        config = MTLBlenderConfig()
        assert config.model_name == "default"
        assert config.mtl_weight == 0.3
        assert config.min_pa == 100

    def test_custom_weight(self) -> None:
        config = MTLBlenderConfig(mtl_weight=0.5)
        assert config.mtl_weight == 0.5
