"""Tests for the ensemble adjuster framework."""

from __future__ import annotations

import pytest

from fantasy_baseball_manager.pipeline.stages.ensemble import (
    EnsembleAdjuster,
    EnsembleConfig,
    WeightedAverageStrategy,
)
from fantasy_baseball_manager.pipeline.stages.prediction_source import PredictionMode
from fantasy_baseball_manager.pipeline.types import PlayerRates
from fantasy_baseball_manager.player.identity import Player

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batter(
    player_id: str = "fg123",
    year: int = 2024,
    rates: dict[str, float] | None = None,
) -> PlayerRates:
    return PlayerRates(
        player_id=player_id,
        name="Test Batter",
        year=year,
        age=28,
        rates=rates or {
            "hr": 0.040,
            "so": 0.200,
            "bb": 0.100,
            "singles": 0.150,
        },
        opportunities=500.0,
        metadata={"pa_per_year": [500.0]},
        player=Player(yahoo_id="", fangraphs_id=player_id, mlbam_id="mlbam123", name="Test Batter"),
    )


class FakePredictionSource:
    """In-memory PredictionSource for testing."""

    def __init__(
        self,
        name: str,
        prediction_mode: PredictionMode,
        predictions: dict[str, float] | None = None,
    ) -> None:
        self._name = name
        self._prediction_mode = prediction_mode
        self._predictions = predictions
        self.ensure_ready_calls: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def prediction_mode(self) -> PredictionMode:
        return self._prediction_mode

    def predict(self, player: PlayerRates) -> dict[str, float] | None:
        return self._predictions

    def ensure_ready(self, year: int) -> None:
        self.ensure_ready_calls.append(year)


# ---------------------------------------------------------------------------
# WeightedAverageStrategy
# ---------------------------------------------------------------------------


class TestWeightedAverageStrategy:
    def test_pure_marcel(self) -> None:
        """When only Marcel weight exists, returns baseline."""
        strategy = WeightedAverageStrategy()
        result = strategy.blend(
            baseline_rate=0.040,
            predictions={},
            weights={"marcel": 1.0},
        )
        assert result == pytest.approx(0.040)

    def test_single_model(self) -> None:
        """Single model blended with Marcel."""
        strategy = WeightedAverageStrategy()
        result = strategy.blend(
            baseline_rate=0.040,
            predictions={"mtl": 0.050},
            weights={"marcel": 0.7, "mtl": 0.3},
        )
        expected = 0.7 * 0.040 + 0.3 * 0.050
        assert result == pytest.approx(expected)

    def test_two_models(self) -> None:
        """Two models blended with Marcel."""
        strategy = WeightedAverageStrategy()
        result = strategy.blend(
            baseline_rate=0.040,
            predictions={"mtl": 0.050, "contextual": 0.060},
            weights={"marcel": 0.5, "mtl": 0.25, "contextual": 0.25},
        )
        expected = 0.5 * 0.040 + 0.25 * 0.050 + 0.25 * 0.060
        assert result == pytest.approx(expected)

    def test_renormalization(self) -> None:
        """Weights are renormalized when they don't sum to 1."""
        strategy = WeightedAverageStrategy()
        # Weights sum to 0.75 → renormalize
        result = strategy.blend(
            baseline_rate=0.040,
            predictions={"mtl": 0.060},
            weights={"marcel": 0.5, "mtl": 0.25},
        )
        # After renormalization: marcel=0.5/0.75, mtl=0.25/0.75
        expected = (0.5 * 0.040 + 0.25 * 0.060) / 0.75
        assert result == pytest.approx(expected)

    def test_zero_total_weight_returns_baseline(self) -> None:
        """When all weights are zero, return baseline."""
        strategy = WeightedAverageStrategy()
        result = strategy.blend(
            baseline_rate=0.040,
            predictions={"mtl": 0.060},
            weights={"marcel": 0.0, "mtl": 0.0},
        )
        assert result == pytest.approx(0.040)


# ---------------------------------------------------------------------------
# EnsembleConfig
# ---------------------------------------------------------------------------


class TestEnsembleConfig:
    def test_defaults(self) -> None:
        config = EnsembleConfig()
        assert config.default_weights == {"marcel": 1.0}
        assert config.weights == {}

    def test_custom_default_weights(self) -> None:
        config = EnsembleConfig(
            default_weights={"marcel": 0.5, "mtl": 0.25, "contextual": 0.25},
        )
        assert config.default_weights["mtl"] == 0.25

    def test_stat_specific_weights(self) -> None:
        config = EnsembleConfig(
            default_weights={"marcel": 0.5, "mtl": 0.25, "contextual": 0.25},
            weights={"hr": {"marcel": 0.4, "mtl": 0.3, "contextual": 0.3}},
        )
        assert config.weights["hr"]["mtl"] == 0.3

    def test_frozen(self) -> None:
        config = EnsembleConfig()
        with pytest.raises(AttributeError):
            config.default_weights = {"marcel": 0.5}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Weight resolution
# ---------------------------------------------------------------------------


class TestWeightResolution:
    def test_stat_specific_override(self) -> None:
        """Stat-specific weights take priority over defaults."""
        config = EnsembleConfig(
            default_weights={"marcel": 0.5, "mtl": 0.5},
            weights={"hr": {"marcel": 0.3, "mtl": 0.7}},
        )
        source = FakePredictionSource("mtl", PredictionMode.RATE, {"hr": 0.05})
        adjuster = EnsembleAdjuster(
            sources=[source],
            config=config,
        )
        batter = _make_batter(rates={"hr": 0.040})
        result = adjuster.adjust([batter])
        # hr uses stat-specific weights: marcel=0.3, mtl=0.7
        expected = 0.3 * 0.040 + 0.7 * 0.05
        assert result[0].rates["hr"] == pytest.approx(expected)

    def test_missing_source_excluded(self) -> None:
        """When a source returns None for a stat, its weight is excluded."""
        config = EnsembleConfig(
            default_weights={"marcel": 0.5, "mtl": 0.25, "contextual": 0.25},
        )
        # Only mtl returns predictions; contextual returns None
        mtl = FakePredictionSource("mtl", PredictionMode.RATE, {"hr": 0.060})
        ctx = FakePredictionSource("contextual", PredictionMode.RATE, None)
        adjuster = EnsembleAdjuster(
            sources=[mtl, ctx],
            config=config,
        )
        batter = _make_batter(rates={"hr": 0.040})
        result = adjuster.adjust([batter])
        # Contextual is excluded; renormalize marcel=0.5, mtl=0.25
        expected = (0.5 * 0.040 + 0.25 * 0.060) / (0.5 + 0.25)
        assert result[0].rates["hr"] == pytest.approx(expected)

    def test_implicit_marcel_complement(self) -> None:
        """When no explicit Marcel weight, it gets 1 - sum(model_weights)."""
        config = EnsembleConfig(
            default_weights={"mtl": 0.3},
        )
        source = FakePredictionSource("mtl", PredictionMode.RATE, {"hr": 0.060})
        adjuster = EnsembleAdjuster(
            sources=[source],
            config=config,
        )
        batter = _make_batter(rates={"hr": 0.040})
        result = adjuster.adjust([batter])
        # Marcel gets 1 - 0.3 = 0.7 implicitly
        expected = 0.7 * 0.040 + 0.3 * 0.060
        assert result[0].rates["hr"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# EnsembleAdjuster
# ---------------------------------------------------------------------------


class TestEnsembleAdjuster:
    def test_empty_list(self) -> None:
        adjuster = EnsembleAdjuster(sources=[])
        assert adjuster.adjust([]) == []

    def test_all_none_passthrough(self) -> None:
        """When all sources return None, player passes through unchanged."""
        source = FakePredictionSource("mtl", PredictionMode.RATE, None)
        adjuster = EnsembleAdjuster(
            sources=[source],
            config=EnsembleConfig(default_weights={"marcel": 0.7, "mtl": 0.3}),
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates

    def test_single_rate_source(self) -> None:
        """Single RATE source blended with Marcel."""
        source = FakePredictionSource("mtl", PredictionMode.RATE, {
            "hr": 0.050, "so": 0.180, "bb": 0.110, "singles": 0.160,
        })
        config = EnsembleConfig(
            default_weights={"marcel": 0.7, "mtl": 0.3},
        )
        adjuster = EnsembleAdjuster(sources=[source], config=config)
        batter = _make_batter()
        result = adjuster.adjust([batter])

        expected_hr = 0.7 * 0.040 + 0.3 * 0.050
        assert result[0].rates["hr"] == pytest.approx(expected_hr)

    def test_two_rate_sources_partial_coverage(self) -> None:
        """Two RATE sources where one covers fewer stats."""
        mtl = FakePredictionSource("mtl", PredictionMode.RATE, {
            "hr": 0.050, "so": 0.180, "bb": 0.110, "singles": 0.160,
        })
        # Contextual only covers hr and so
        ctx = FakePredictionSource("contextual", PredictionMode.RATE, {
            "hr": 0.060, "so": 0.170,
        })
        config = EnsembleConfig(
            default_weights={"marcel": 0.5, "mtl": 0.25, "contextual": 0.25},
        )
        adjuster = EnsembleAdjuster(sources=[mtl, ctx], config=config)
        batter = _make_batter()
        result = adjuster.adjust([batter])

        # HR: all three sources contribute
        expected_hr = 0.5 * 0.040 + 0.25 * 0.050 + 0.25 * 0.060
        assert result[0].rates["hr"] == pytest.approx(expected_hr)

        # bb: contextual doesn't cover → renormalize marcel(0.5) + mtl(0.25)
        expected_bb = (0.5 * 0.100 + 0.25 * 0.110) / (0.5 + 0.25)
        assert result[0].rates["bb"] == pytest.approx(expected_bb)

    def test_residual_source(self) -> None:
        """RESIDUAL source adds to rates after RATE blending."""
        gb = FakePredictionSource("gb_residual", PredictionMode.RESIDUAL, {
            "hr": 0.005,  # Add 0.005 to hr rate
        })
        adjuster = EnsembleAdjuster(
            sources=[gb],
            config=EnsembleConfig(default_weights={"marcel": 1.0}),
        )
        batter = _make_batter()
        result = adjuster.adjust([batter])
        # Pure Marcel + residual: 0.040 + 0.005 = 0.045
        assert result[0].rates["hr"] == pytest.approx(0.045)

    def test_mixed_rate_and_residual(self) -> None:
        """RATE blending first, then RESIDUAL applied on top."""
        mtl = FakePredictionSource("mtl", PredictionMode.RATE, {"hr": 0.050})
        gb = FakePredictionSource("gb_residual", PredictionMode.RESIDUAL, {"hr": 0.005})
        config = EnsembleConfig(
            default_weights={"marcel": 0.7, "mtl": 0.3},
        )
        adjuster = EnsembleAdjuster(sources=[mtl, gb], config=config)
        batter = _make_batter()
        result = adjuster.adjust([batter])

        blended_hr = 0.7 * 0.040 + 0.3 * 0.050
        expected_hr = blended_hr + 0.005
        assert result[0].rates["hr"] == pytest.approx(expected_hr)

    def test_metadata_accumulation(self) -> None:
        """Ensemble metadata is stored in player metadata."""
        mtl = FakePredictionSource("mtl", PredictionMode.RATE, {"hr": 0.050})
        gb = FakePredictionSource("gb_residual", PredictionMode.RESIDUAL, {"hr": 0.005})
        config = EnsembleConfig(
            default_weights={"marcel": 0.7, "mtl": 0.3},
        )
        adjuster = EnsembleAdjuster(sources=[mtl, gb], config=config)
        batter = _make_batter()
        result = adjuster.adjust([batter])

        assert result[0].metadata["ensemble_blended"] is True
        assert "mtl" in result[0].metadata["ensemble_sources"]
        assert "gb_residual" in result[0].metadata["ensemble_residual_sources"]

    def test_ensure_ready_called_with_year(self) -> None:
        """ensure_ready is called on each source with the player year."""
        source = FakePredictionSource("mtl", PredictionMode.RATE, None)
        adjuster = EnsembleAdjuster(sources=[source])
        batter = _make_batter(year=2025)
        adjuster.adjust([batter])
        assert source.ensure_ready_calls == [2025]

    def test_no_sources_returns_unchanged(self) -> None:
        """With no sources, players pass through unchanged."""
        adjuster = EnsembleAdjuster(sources=[])
        batter = _make_batter()
        result = adjuster.adjust([batter])
        assert result[0].rates == batter.rates

    def test_preserves_non_covered_stats(self) -> None:
        """Stats not covered by any source remain at Marcel baseline."""
        source = FakePredictionSource("mtl", PredictionMode.RATE, {"hr": 0.050})
        config = EnsembleConfig(default_weights={"marcel": 0.7, "mtl": 0.3})
        adjuster = EnsembleAdjuster(sources=[source], config=config)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        # 'singles' not in mtl predictions → pure Marcel (renormalized: marcel-only = 1.0)
        assert result[0].rates["singles"] == pytest.approx(0.150)

    def test_custom_strategy(self) -> None:
        """Custom blending strategy is used."""

        class DoubleStrategy:
            def blend(
                self,
                baseline_rate: float,
                predictions: dict[str, float],
                weights: dict[str, float],
            ) -> float:
                return baseline_rate * 2.0

        source = FakePredictionSource("mtl", PredictionMode.RATE, {"hr": 0.050})
        config = EnsembleConfig(
            default_weights={"marcel": 0.7, "mtl": 0.3},
            strategy=DoubleStrategy(),
        )
        adjuster = EnsembleAdjuster(sources=[source], config=config)
        batter = _make_batter()
        result = adjuster.adjust([batter])
        # DoubleStrategy doubles the baseline
        assert result[0].rates["hr"] == pytest.approx(0.080)
