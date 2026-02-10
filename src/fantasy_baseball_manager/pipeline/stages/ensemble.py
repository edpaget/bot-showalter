"""Ensemble adjuster for combining multiple prediction sources.

Provides a generic framework for blending predictions from multiple
model types (MTL, contextual, gradient boosting) with configurable
per-stat-per-model weights and pluggable blending strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from fantasy_baseball_manager.pipeline.stages.prediction_source import (
    PredictionMode,
    PredictionSource,
)
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blending strategy
# ---------------------------------------------------------------------------


@runtime_checkable
class BlendingStrategy(Protocol):
    """Strategy for combining rate predictions from multiple sources."""

    def blend(
        self,
        baseline_rate: float,
        predictions: dict[str, float],
        weights: dict[str, float],
    ) -> float: ...


class WeightedAverageStrategy:
    """Weighted average blending: sum(w_i * rate_i) / sum(w_i).

    Marcel baseline is included with key "marcel" in weights.
    """

    def blend(
        self,
        baseline_rate: float,
        predictions: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        total_weight = 0.0
        weighted_sum = 0.0

        marcel_weight = weights.get("marcel", 0.0)
        if marcel_weight > 0:
            weighted_sum += marcel_weight * baseline_rate
            total_weight += marcel_weight

        for source_name, rate in predictions.items():
            w = weights.get(source_name, 0.0)
            if w > 0:
                weighted_sum += w * rate
                total_weight += w

        if total_weight == 0:
            return baseline_rate

        return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleConfig:
    """Configuration for the ensemble adjuster.

    Attributes:
        default_weights: Default weights for all stats. Keys are source
            names plus "marcel" for the baseline. If "marcel" is not
            present, it gets ``1.0 - sum(model_weights)``.
        weights: Per-stat weight overrides. Keys are stat names, values
            are weight dicts that override default_weights for that stat.
        strategy: Blending strategy for RATE sources. Defaults to
            WeightedAverageStrategy.
    """

    default_weights: dict[str, float] = field(default_factory=lambda: {"marcel": 1.0})
    weights: dict[str, dict[str, float]] = field(default_factory=dict)
    strategy: BlendingStrategy = field(default_factory=WeightedAverageStrategy)


# ---------------------------------------------------------------------------
# EnsembleAdjuster
# ---------------------------------------------------------------------------


@dataclass
class EnsembleAdjuster:
    """Combines multiple prediction sources with configurable weights.

    Processing order per player:
    1. Call predict() on each source
    2. For each stat, blend RATE predictions with Marcel baseline
    3. Apply RESIDUAL corrections additively on top

    Implements the RateAdjuster protocol.
    """

    sources: list[PredictionSource]
    config: EnsembleConfig = field(default_factory=EnsembleConfig)

    def adjust(self, players: list[PlayerRates]) -> list[PlayerRates]:
        if not players:
            return []

        year = players[0].year
        for source in self.sources:
            source.ensure_ready(year)

        return [self._adjust_player(player) for player in players]

    def _adjust_player(self, player: PlayerRates) -> PlayerRates:
        if not self.sources:
            return player

        rate_predictions: dict[str, dict[str, float]] = {}
        residual_predictions: dict[str, dict[str, float]] = {}

        for source in self.sources:
            preds = source.predict(player)
            if preds is None:
                continue
            if source.prediction_mode == PredictionMode.RATE:
                rate_predictions[source.name] = preds
            else:
                residual_predictions[source.name] = preds

        if not rate_predictions and not residual_predictions:
            return player

        rates = dict(player.rates)

        # Step 1: Blend RATE predictions with Marcel baseline
        if rate_predictions:
            all_stats = set(rates.keys())
            for source_preds in rate_predictions.values():
                all_stats.update(source_preds.keys())

            for stat in all_stats:
                if stat not in rates:
                    continue

                stat_preds: dict[str, float] = {}
                for source_name, preds in rate_predictions.items():
                    if stat in preds:
                        stat_preds[source_name] = preds[stat]

                if not stat_preds:
                    continue

                weights = self._resolve_weights(stat, set(stat_preds.keys()))
                rates[stat] = self.config.strategy.blend(
                    baseline_rate=player.rates[stat],
                    predictions=stat_preds,
                    weights=weights,
                )

        # Step 2: Apply RESIDUAL corrections additively
        for _source_name, preds in residual_predictions.items():
            for stat, adjustment in preds.items():
                if stat in rates:
                    rates[stat] = rates[stat] + adjustment

        # Build metadata
        ensemble_sources: dict[str, dict[str, float]] = {}
        for source_name, preds in rate_predictions.items():
            ensemble_sources[source_name] = preds

        ensemble_residual_sources: dict[str, dict[str, float]] = {}
        for source_name, preds in residual_predictions.items():
            ensemble_residual_sources[source_name] = preds

        metadata: PlayerMetadata = {
            **player.metadata,
            "ensemble_blended": True,
            "ensemble_sources": ensemble_sources,
            "ensemble_residual_sources": ensemble_residual_sources,
        }

        return PlayerRates(
            player_id=player.player_id,
            name=player.name,
            year=player.year,
            age=player.age,
            rates=rates,
            opportunities=player.opportunities,
            metadata=metadata,
            player=player.player,
        )

    def _resolve_weights(
        self,
        stat: str,
        available_sources: set[str],
    ) -> dict[str, float]:
        """Resolve weights for a stat, excluding unavailable sources.

        Uses stat-specific weights if defined, otherwise default_weights.
        If "marcel" is not in the weight dict, it gets the complement
        of the sum of model weights (1 - sum).
        """
        base = self.config.weights.get(stat, self.config.default_weights)

        # Filter to marcel + available sources
        weights: dict[str, float] = {}
        if "marcel" in base:
            weights["marcel"] = base["marcel"]
        for source_name in available_sources:
            if source_name in base:
                weights[source_name] = base[source_name]

        # If marcel not explicitly set, compute as complement
        if "marcel" not in weights:
            model_sum = sum(weights.values())
            weights["marcel"] = max(0.0, 1.0 - model_sum)

        return weights
