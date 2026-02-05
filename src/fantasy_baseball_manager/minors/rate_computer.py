"""MLE-based rate computer for minor league players with limited MLB history."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.minors.features import MLEBatterFeatureExtractor
from fantasy_baseball_manager.minors.persistence import MLEModelStore
from fantasy_baseball_manager.minors.training_data import (
    BATTER_TARGET_STATS,
    AggregatedMiLBStats,
)
from fantasy_baseball_manager.minors.types import MinorLeagueLevel
from fantasy_baseball_manager.pipeline.stages.rate_computers import MarcelRateComputer
from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.minors.data_source import MinorLeagueDataSource
    from fantasy_baseball_manager.minors.model import MLEGradientBoostingModel
    from fantasy_baseball_manager.minors.types import MinorLeagueBatterSeasonStats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLERateComputerConfig:
    """Configuration for MLERateComputer.

    Attributes:
        model_name: Name of the MLE model to load (default: "default")
        min_milb_pa: Minimum MiLB PA required to apply MLE predictions
        mlb_pa_threshold: MLB PA threshold below which MLE is applied
    """

    model_name: str = "default"
    min_milb_pa: int = 200
    mlb_pa_threshold: int = 200


@dataclass
class MLERateComputer:
    """Computes player rates using ML-based Minor League Equivalencies.

    For players with limited MLB history (<mlb_pa_threshold PA), this computer
    uses trained MLE models to translate their minor league stats to MLB
    equivalents and blends them with available MLB data.

    For players with sufficient MLB history, falls back to standard Marcel rates.

    Implements the RateComputer protocol.
    """

    milb_source: MinorLeagueDataSource
    config: MLERateComputerConfig = field(default_factory=MLERateComputerConfig)
    model_store: MLEModelStore = field(default_factory=MLEModelStore)

    # Fallback to Marcel for established MLB players or when MLE unavailable
    _marcel_computer: MarcelRateComputer = field(
        default_factory=MarcelRateComputer, repr=False
    )

    # Lazy-loaded model
    _batter_model: MLEGradientBoostingModel | None = field(
        default=None, init=False, repr=False
    )
    _model_loaded: bool = field(default=False, init=False, repr=False)

    # Cache for MiLB data by year
    _milb_cache: dict[int, dict[str, list[MinorLeagueBatterSeasonStats]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _ensure_model_loaded(self) -> None:
        """Lazy-load trained MLE model on first use."""
        if self._model_loaded:
            return

        if self.model_store.exists(self.config.model_name, "batter"):
            self._batter_model = self.model_store.load(
                self.config.model_name, "batter"
            )
            logger.debug("Loaded MLE batter model: %s", self.config.model_name)
        else:
            logger.warning(
                "MLE batter model %s not found, using Marcel fallback",
                self.config.model_name,
            )

        self._model_loaded = True

    def compute_batting_rates(
        self,
        data_source: StatsDataSource,
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute batting rates using MLE model for players with limited MLB history.

        For players with < mlb_pa_threshold MLB PA:
        1. Get their MiLB stats from the prior year
        2. Use MLE model to predict MLB-equivalent rates
        3. Blend MLE predictions with available MLB data (PA-weighted)

        For players with sufficient MLB history, returns standard Marcel rates.

        Args:
            data_source: Source for historical MLB stats.
            year: Target projection year.
            years_back: Number of years of history to consider.

        Returns:
            List of PlayerRates with rates and metadata.
        """
        self._ensure_model_loaded()

        # Get Marcel rates as baseline
        marcel_rates = self._marcel_computer.compute_batting_rates(
            data_source, year, years_back
        )

        if self._batter_model is None:
            logger.debug("No MLE batter model available, using Marcel rates")
            return marcel_rates

        # Get MiLB stats from prior year for MLE predictions
        milb_year = year - 1
        milb_by_player = self._get_milb_stats_by_player(milb_year)

        # Calculate total MLB PA per player from recent years
        mlb_pa_by_player = self._calculate_mlb_pa(data_source, year, years_back)

        feature_extractor = MLEBatterFeatureExtractor(min_pa=50)

        result: list[PlayerRates] = []
        mle_count = 0

        for marcel_player in marcel_rates:
            player_id = marcel_player.player_id
            total_mlb_pa = mlb_pa_by_player.get(player_id, 0)

            # Check if player needs MLE (limited MLB history)
            if total_mlb_pa >= self.config.mlb_pa_threshold:
                # Established MLB player - use Marcel rates
                result.append(marcel_player)
                continue

            # Check if we have MiLB stats for this player
            milb_seasons = milb_by_player.get(player_id, [])
            if not milb_seasons:
                # No MiLB data - use Marcel rates
                result.append(marcel_player)
                continue

            # Aggregate MiLB stats across levels
            aggregated = AggregatedMiLBStats.from_seasons(milb_seasons)
            if aggregated is None or aggregated.total_pa < self.config.min_milb_pa:
                # Insufficient MiLB PA - use Marcel rates
                result.append(marcel_player)
                continue

            # Check level (only AAA/AA for now)
            if aggregated.highest_level not in (
                MinorLeagueLevel.AAA,
                MinorLeagueLevel.AA,
            ):
                result.append(marcel_player)
                continue

            # Extract features from MiLB stats
            features = feature_extractor.extract(aggregated)
            if features is None:
                result.append(marcel_player)
                continue

            # Get MLE predictions
            mle_predictions = self._batter_model.predict_rates(features)

            # Blend MLE with Marcel rates
            blended_rates = self._blend_rates(
                marcel_rates=marcel_player.rates,
                marcel_pa=total_mlb_pa,
                mle_rates=mle_predictions,
                mle_pa=aggregated.total_pa,
            )

            result.append(
                PlayerRates(
                    player_id=marcel_player.player_id,
                    name=marcel_player.name,
                    year=year,
                    age=marcel_player.age,
                    rates=blended_rates,
                    opportunities=marcel_player.opportunities,
                    metadata={
                        **marcel_player.metadata,
                        "mle_applied": True,
                        "mle_source_level": aggregated.highest_level.display_name,
                        "mle_source_pa": aggregated.total_pa,
                        "marcel_rates": marcel_player.rates,
                        "mle_rates": mle_predictions,
                    },
                )
            )
            mle_count += 1

        logger.info(
            "MLE rate computer: %d/%d batters used MLE predictions",
            mle_count,
            len(result),
        )
        return result

    def compute_pitching_rates(
        self,
        data_source: StatsDataSource,
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute pitching rates using Marcel (MLE pitchers not yet implemented).

        Args:
            data_source: Source for historical MLB stats.
            year: Target projection year.
            years_back: Number of years of history to consider.

        Returns:
            List of PlayerRates from Marcel.
        """
        # MLE for pitchers not yet implemented - fall back to Marcel
        return self._marcel_computer.compute_pitching_rates(
            data_source, year, years_back
        )

    def _get_milb_stats_by_player(
        self, year: int
    ) -> dict[str, list[MinorLeagueBatterSeasonStats]]:
        """Get MiLB stats grouped by player ID for a year."""
        if year in self._milb_cache:
            return self._milb_cache[year]

        all_stats = self.milb_source.batting_stats_all_levels(year)
        by_player: dict[str, list[MinorLeagueBatterSeasonStats]] = {}
        for stat in all_stats:
            if stat.player_id not in by_player:
                by_player[stat.player_id] = []
            by_player[stat.player_id].append(stat)

        self._milb_cache[year] = by_player
        return by_player

    def _calculate_mlb_pa(
        self,
        data_source: StatsDataSource,
        year: int,
        years_back: int,
    ) -> dict[str, int]:
        """Calculate total MLB PA per player from recent years."""
        pa_by_player: dict[str, int] = {}

        for y in range(year - years_back, year):
            for player in data_source.batting_stats(y):
                if player.player_id not in pa_by_player:
                    pa_by_player[player.player_id] = 0
                pa_by_player[player.player_id] += player.pa

        return pa_by_player

    def _blend_rates(
        self,
        marcel_rates: dict[str, float],
        marcel_pa: int,
        mle_rates: dict[str, float],
        mle_pa: int,
    ) -> dict[str, float]:
        """Blend Marcel and MLE rates using PA-weighted average.

        For stats that MLE predicts, uses PA-weighted blend.
        For other stats, uses Marcel rates.

        Args:
            marcel_rates: Rates from Marcel projection
            marcel_pa: MLB PA used for Marcel
            mle_rates: Rates from MLE prediction
            mle_pa: MiLB PA used for MLE

        Returns:
            Blended rates dictionary
        """
        total_pa = marcel_pa + mle_pa
        if total_pa == 0:
            return marcel_rates.copy()

        blended: dict[str, float] = {}

        for stat, marcel_val in marcel_rates.items():
            if stat in mle_rates:
                # PA-weighted blend for stats MLE predicts
                mle_val = mle_rates[stat]
                blended[stat] = (
                    marcel_pa * marcel_val + mle_pa * mle_val
                ) / total_pa
            else:
                # Use Marcel for stats MLE doesn't predict
                blended[stat] = marcel_val

        return blended
