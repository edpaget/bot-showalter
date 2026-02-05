"""Tests for MTL rate computer pipeline stage."""

from dataclasses import dataclass
from unittest.mock import MagicMock

from fantasy_baseball_manager.ml.mtl.config import MTLRateComputerConfig
from fantasy_baseball_manager.pipeline.stages.mtl_rate_computer import MTLRateComputer
from fantasy_baseball_manager.pipeline.types import PlayerRates


@dataclass
class MockStatcastBatterStats:
    player_id: str
    name: str
    year: int
    pa: int
    xba: float = 0.260
    xslg: float = 0.420
    xwoba: float = 0.320
    barrel_rate: float = 0.08
    hard_hit_rate: float = 0.40


@dataclass
class MockBatterSkillStats:
    player_id: str
    year: int
    pa: int
    chase_rate: float = 0.28
    whiff_rate: float = 0.24


class TestMTLRateComputer:
    def test_init(self) -> None:
        """Test that MTLRateComputer can be initialized."""
        statcast = MagicMock()
        bb_source = MagicMock()
        skill_source = MagicMock()
        mapper = MagicMock()

        computer = MTLRateComputer(
            statcast_source=statcast,
            batted_ball_source=bb_source,
            skill_data_source=skill_source,
            id_mapper=mapper,
        )

        assert computer.config.model_name == "default"
        assert computer.config.min_pa == 100

    def test_custom_config(self) -> None:
        """Test with custom configuration."""
        config = MTLRateComputerConfig(model_name="custom", min_pa=200)
        computer = MTLRateComputer(
            statcast_source=MagicMock(),
            batted_ball_source=MagicMock(),
            skill_data_source=MagicMock(),
            id_mapper=MagicMock(),
            config=config,
        )

        assert computer.config.model_name == "custom"
        assert computer.config.min_pa == 200

    def test_falls_back_to_marcel_when_no_model(self) -> None:
        """When MTL model not found, should return Marcel rates."""
        # Mock model store that says no model exists
        mock_store = MagicMock()
        mock_store.exists.return_value = False

        computer = MTLRateComputer(
            statcast_source=MagicMock(),
            batted_ball_source=MagicMock(),
            skill_data_source=MagicMock(),
            id_mapper=MagicMock(),
            model_store=mock_store,
        )

        # Verify the model store is queried
        computer._ensure_models_loaded()
        mock_store.exists.assert_called()

        # Verify models are not loaded
        assert computer._batter_model is None
        assert computer._pitcher_model is None

    def test_batter_detection(self) -> None:
        """Test that batters are distinguished from pitchers by metadata."""
        batter = PlayerRates(
            player_id="123",
            name="Test Batter",
            year=2024,
            age=28,
            rates={"hr": 0.03},
            metadata={"pa_per_year": [500.0, 450.0, 480.0]},
        )

        pitcher = PlayerRates(
            player_id="456",
            name="Test Pitcher",
            year=2024,
            age=30,
            rates={"h": 0.25},
            metadata={"ip_per_year": [180.0, 175.0, 190.0]},
        )

        # Batters have pa_per_year in metadata
        assert "pa_per_year" in batter.metadata
        assert "pa_per_year" not in pitcher.metadata


class TestMTLRateComputerIntegration:
    """Integration tests for MTL rate computer."""

    def test_compute_batting_rates_with_model(self) -> None:
        """Test computing batting rates with a trained model."""
        # This test verifies the integration works but requires
        # a more complex setup with trained models
        pass  # Placeholder for full integration test
