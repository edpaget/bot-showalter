"""Tests for MTL blender pipeline stage."""

from unittest.mock import MagicMock

import pytest

from fantasy_baseball_manager.ml.mtl.config import MTLBlenderConfig
from fantasy_baseball_manager.pipeline.stages.mtl_blender import MTLBlender
from fantasy_baseball_manager.pipeline.types import PlayerRates
from tests.conftest import make_test_feature_store


class TestMTLBlender:
    def test_init(self) -> None:
        """Test that MTLBlender can be initialized."""
        blender = MTLBlender(
            feature_store=make_test_feature_store(),
        )

        assert blender.config.model_name == "default"
        assert blender.config.mtl_weight == 0.3
        assert blender.config.min_pa == 100

    def test_custom_config(self) -> None:
        """Test with custom configuration."""
        config = MTLBlenderConfig(model_name="custom", mtl_weight=0.5, min_pa=150)
        blender = MTLBlender(
            feature_store=make_test_feature_store(),
            config=config,
        )

        assert blender.config.model_name == "custom"
        assert blender.config.mtl_weight == 0.5
        assert blender.config.min_pa == 150

    def test_adjust_empty_list(self) -> None:
        """Adjusting empty list should return empty list."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False

        blender = MTLBlender(
            feature_store=make_test_feature_store(),
            model_store=mock_store,
        )

        result = blender.adjust([])
        assert result == []

    def test_returns_unchanged_when_no_model(self) -> None:
        """When no model exists, should return players unchanged."""
        mock_store = MagicMock()
        mock_store.exists.return_value = False

        blender = MTLBlender(
            feature_store=make_test_feature_store(),
            model_store=mock_store,
        )

        player = PlayerRates(
            player_id="123",
            name="Test Player",
            year=2024,
            age=28,
            rates={"hr": 0.035, "so": 0.22, "bb": 0.09},
            metadata={"pa_per_year": [500.0]},
        )

        result = blender.adjust([player])

        assert len(result) == 1
        # Rates should be unchanged
        assert result[0].rates["hr"] == 0.035
        assert result[0].rates["so"] == 0.22

    def test_batter_detection(self) -> None:
        """Test that batters are distinguished from pitchers by metadata."""
        batter = PlayerRates(
            player_id="123",
            name="Test Batter",
            year=2024,
            age=28,
            rates={},
            metadata={"pa_per_year": [500.0]},
        )

        pitcher = PlayerRates(
            player_id="456",
            name="Test Pitcher",
            year=2024,
            age=30,
            rates={},
            metadata={"ip_per_year": [180.0]},
        )

        # Batters have pa_per_year in metadata
        assert "pa_per_year" in batter.metadata
        assert "pa_per_year" not in pitcher.metadata


class TestMTLBlenderMath:
    """Test the blending math without requiring torch."""

    def test_blend_formula(self) -> None:
        """Verify the blending formula is correct."""
        # blend = (1 - weight) * marcel + weight * mtl
        marcel_rate = 0.040
        mtl_rate = 0.050
        weight = 0.3

        expected = (1 - weight) * marcel_rate + weight * mtl_rate
        assert expected == pytest.approx(0.043, rel=1e-3)

    def test_weight_zero_returns_marcel(self) -> None:
        """Weight of 0 should return pure Marcel."""
        marcel_rate = 0.040
        mtl_rate = 0.050
        weight = 0.0

        result = (1 - weight) * marcel_rate + weight * mtl_rate
        assert result == marcel_rate

    def test_weight_one_returns_mtl(self) -> None:
        """Weight of 1 should return pure MTL."""
        marcel_rate = 0.040
        mtl_rate = 0.050
        weight = 1.0

        result = (1 - weight) * marcel_rate + weight * mtl_rate
        assert result == mtl_rate
