"""MLE-based rate computer for minor league players with limited MLB history."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fantasy_baseball_manager.context import new_context
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS
from fantasy_baseball_manager.minors.features import MLEBatterFeatureExtractor
from fantasy_baseball_manager.minors.persistence import MLEModelStore
from fantasy_baseball_manager.minors.training_data import (
    AggregatedMiLBStats,
)
from fantasy_baseball_manager.minors.types import MinorLeagueLevel
from fantasy_baseball_manager.pipeline.stages.rate_computers import MarcelRateComputer
from fantasy_baseball_manager.pipeline.types import PlayerRates

if TYPE_CHECKING:
    from fantasy_baseball_manager.data.protocol import DataSource
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats
    from fantasy_baseball_manager.minors.model import MLEGradientBoostingModel
    from fantasy_baseball_manager.minors.types import MinorLeagueBatterSeasonStats
    from fantasy_baseball_manager.pipeline.protocols import RateComputer
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

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

    milb_source: DataSource[MinorLeagueBatterSeasonStats]
    config: MLERateComputerConfig = field(default_factory=MLERateComputerConfig)
    model_store: MLEModelStore = field(default_factory=MLEModelStore)

    # Fallback to Marcel for established MLB players or when MLE unavailable
    _marcel_computer: MarcelRateComputer = field(default_factory=MarcelRateComputer, repr=False)

    # Lazy-loaded model
    _batter_model: MLEGradientBoostingModel | None = field(default=None, init=False, repr=False)
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
            self._batter_model = self.model_store.load(self.config.model_name, "batter")
            logger.debug("Loaded MLE batter model: %s", self.config.model_name)
        else:
            logger.warning(
                "MLE batter model %s not found, using Marcel fallback",
                self.config.model_name,
            )

        self._model_loaded = True

    def compute_batting_rates(
        self,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute batting rates using MLE model for players with limited MLB history.

        For players with < mlb_pa_threshold MLB PA:
        1. Get their MiLB stats from the prior year
        2. Use MLE model to predict MLB-equivalent rates
        3. Blend MLE predictions with available MLB data (PA-weighted)

        For players with sufficient MLB history, returns standard Marcel rates.
        """
        self._ensure_model_loaded()

        # Get Marcel rates as baseline
        marcel_rates = self._marcel_computer.compute_batting_rates(
            batting_source, team_batting_source, year, years_back
        )

        if self._batter_model is None:
            logger.debug("No MLE batter model available, using Marcel rates")
            return marcel_rates

        # Get MiLB stats from prior year for MLE predictions
        milb_year = year - 1
        milb_by_player = self._get_milb_stats_by_player(milb_year)

        # Calculate total MLB PA per player from recent years
        mlb_pa_by_player = self._calculate_mlb_pa(batting_source, year, years_back)

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
        pitching_source: DataSource[PitchingSeasonStats],
        team_pitching_source: DataSource[PitchingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute pitching rates using Marcel (MLE pitchers not yet implemented)."""
        # MLE for pitchers not yet implemented - fall back to Marcel
        return self._marcel_computer.compute_pitching_rates(pitching_source, team_pitching_source, year, years_back)

    def _get_milb_stats_by_player(self, year: int) -> dict[str, list[MinorLeagueBatterSeasonStats]]:
        """Get MiLB stats grouped by player ID for a year."""
        if year in self._milb_cache:
            return self._milb_cache[year]

        with new_context(year=year):
            result = self.milb_source(ALL_PLAYERS)
        all_stats = result.unwrap() if result.is_ok() else []
        by_player: dict[str, list[MinorLeagueBatterSeasonStats]] = {}
        for stat in all_stats:
            if stat.player_id not in by_player:
                by_player[stat.player_id] = []
            by_player[stat.player_id].append(stat)

        self._milb_cache[year] = by_player
        return by_player

    def _calculate_mlb_pa(
        self,
        batting_source: DataSource[BattingSeasonStats],
        year: int,
        years_back: int,
    ) -> dict[str, int]:
        """Calculate total MLB PA per player from recent years."""
        pa_by_player: dict[str, int] = {}

        for y in range(year - years_back, year):
            with new_context(year=y):
                result = batting_source(ALL_PLAYERS)
                if result.is_ok():
                    for player in result.unwrap():
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
        """Blend Marcel and MLE rates using PA-weighted average."""
        total_pa = marcel_pa + mle_pa
        if total_pa == 0:
            return marcel_rates.copy()

        blended: dict[str, float] = {}

        for stat, marcel_val in marcel_rates.items():
            if stat in mle_rates:
                # PA-weighted blend for stats MLE predicts
                mle_val = mle_rates[stat]
                blended[stat] = (marcel_pa * marcel_val + mle_pa * mle_val) / total_pa
            else:
                # Use Marcel for stats MLE doesn't predict
                blended[stat] = marcel_val

        return blended


@dataclass
class MLEAugmentedRateComputer:
    """Wraps any rate computer and augments with MLE for rookies.

    This allows using MLE predictions for players with limited MLB history
    while keeping the full power of advanced rate computers (like
    StatSpecificRegressionRateComputer) for established players.

    For players with < mlb_pa_threshold career MLB PA:
    - Uses MLE model to predict rates from MiLB stats
    - Blends MLE predictions with delegate rates (PA-weighted)

    For players with sufficient MLB history:
    - Uses delegate rate computer unchanged

    Implements the RateComputer protocol.
    """

    delegate: RateComputer
    milb_source: DataSource[MinorLeagueBatterSeasonStats]
    id_mapper: PlayerIdMapper  # Maps FanGraphs IDs to MLBAM IDs
    config: MLERateComputerConfig = field(default_factory=MLERateComputerConfig)
    model_store: MLEModelStore = field(default_factory=MLEModelStore)

    # Lazy-loaded model
    _batter_model: MLEGradientBoostingModel | None = field(default=None, init=False, repr=False)
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
            self._batter_model = self.model_store.load(self.config.model_name, "batter")
            logger.debug("Loaded MLE batter model: %s", self.config.model_name)
        else:
            logger.warning(
                "MLE batter model %s not found, MLE augmentation disabled",
                self.config.model_name,
            )

        self._model_loaded = True

    def compute_batting_rates(
        self,
        batting_source: DataSource[BattingSeasonStats],
        team_batting_source: DataSource[BattingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute batting rates, augmenting rookies with MLE predictions."""
        self._ensure_model_loaded()

        # Get base rates from delegate
        delegate_rates = self.delegate.compute_batting_rates(batting_source, team_batting_source, year, years_back)

        if self._batter_model is None:
            logger.debug("No MLE batter model available, returning delegate rates")
            return delegate_rates

        # Get MiLB stats from prior year for MLE predictions
        milb_year = year - 1
        milb_by_player = self._get_milb_stats_by_player(milb_year)

        # Calculate total MLB PA per player from recent years
        mlb_pa_by_player = self._calculate_mlb_pa(batting_source, year, years_back)

        feature_extractor = MLEBatterFeatureExtractor(min_pa=50)

        result: list[PlayerRates] = []
        mle_count = 0

        for player_rates in delegate_rates:
            player_id = player_rates.player_id  # FanGraphs ID
            total_mlb_pa = mlb_pa_by_player.get(player_id, 0)

            # Check if player needs MLE (limited MLB history)
            if total_mlb_pa >= self.config.mlb_pa_threshold:
                # Established MLB player - use delegate rates
                result.append(player_rates)
                continue

            # Convert FanGraphs ID to MLBAM ID for MiLB lookup
            mlbam_id = self.id_mapper.fangraphs_to_mlbam(player_id)
            if mlbam_id is None:
                # Can't map ID - use delegate rates
                result.append(player_rates)
                continue

            # Check if we have MiLB stats for this player
            milb_seasons = milb_by_player.get(mlbam_id, [])
            if not milb_seasons:
                # No MiLB data - use delegate rates
                result.append(player_rates)
                continue

            # Aggregate MiLB stats across levels
            aggregated = AggregatedMiLBStats.from_seasons(milb_seasons)
            if aggregated is None or aggregated.total_pa < self.config.min_milb_pa:
                # Insufficient MiLB PA - use delegate rates
                result.append(player_rates)
                continue

            # Check level (only AAA/AA for now)
            if aggregated.highest_level not in (
                MinorLeagueLevel.AAA,
                MinorLeagueLevel.AA,
            ):
                result.append(player_rates)
                continue

            # Extract features from MiLB stats
            features = feature_extractor.extract(aggregated)
            if features is None:
                result.append(player_rates)
                continue

            # Get MLE predictions
            mle_predictions = self._batter_model.predict_rates(features)

            # Blend MLE with delegate rates
            blended_rates = self._blend_rates(
                delegate_rates=player_rates.rates,
                delegate_pa=total_mlb_pa,
                mle_rates=mle_predictions,
                mle_pa=aggregated.total_pa,
            )

            result.append(
                PlayerRates(
                    player_id=player_rates.player_id,
                    name=player_rates.name,
                    year=year,
                    age=player_rates.age,
                    rates=blended_rates,
                    opportunities=player_rates.opportunities,
                    metadata={
                        **player_rates.metadata,
                        "mle_augmented": True,
                        "mle_source_level": aggregated.highest_level.display_name,
                        "mle_source_pa": aggregated.total_pa,
                        "delegate_rates": player_rates.rates,
                        "mle_rates": mle_predictions,
                    },
                )
            )
            mle_count += 1

        logger.info(
            "MLE augmented rate computer: %d/%d batters used MLE predictions",
            mle_count,
            len(result),
        )
        return result

    def compute_pitching_rates(
        self,
        pitching_source: DataSource[PitchingSeasonStats],
        team_pitching_source: DataSource[PitchingSeasonStats],
        year: int,
        years_back: int,
    ) -> list[PlayerRates]:
        """Compute pitching rates (MLE pitchers not yet implemented)."""
        # MLE for pitchers not yet implemented - delegate unchanged
        return self.delegate.compute_pitching_rates(pitching_source, team_pitching_source, year, years_back)

    def _get_milb_stats_by_player(self, year: int) -> dict[str, list[MinorLeagueBatterSeasonStats]]:
        """Get MiLB stats grouped by player ID for a year."""
        if year in self._milb_cache:
            return self._milb_cache[year]

        with new_context(year=year):
            result = self.milb_source(ALL_PLAYERS)
        all_stats = result.unwrap() if result.is_ok() else []
        by_player: dict[str, list[MinorLeagueBatterSeasonStats]] = {}
        for stat in all_stats:
            if stat.player_id not in by_player:
                by_player[stat.player_id] = []
            by_player[stat.player_id].append(stat)

        self._milb_cache[year] = by_player
        return by_player

    def _calculate_mlb_pa(
        self,
        batting_source: DataSource[BattingSeasonStats],
        year: int,
        years_back: int,
    ) -> dict[str, int]:
        """Calculate total MLB PA per player from recent years."""
        pa_by_player: dict[str, int] = {}

        for y in range(year - years_back, year):
            with new_context(year=y):
                result = batting_source(ALL_PLAYERS)
                if result.is_ok():
                    for player in result.unwrap():
                        if player.player_id not in pa_by_player:
                            pa_by_player[player.player_id] = 0
                        pa_by_player[player.player_id] += player.pa

        return pa_by_player

    def _blend_rates(
        self,
        delegate_rates: dict[str, float],
        delegate_pa: int,
        mle_rates: dict[str, float],
        mle_pa: int,
    ) -> dict[str, float]:
        """Blend delegate and MLE rates using PA-weighted average."""
        total_pa = delegate_pa + mle_pa
        if total_pa == 0:
            return delegate_rates.copy()

        blended: dict[str, float] = {}

        for stat, delegate_val in delegate_rates.items():
            if stat in mle_rates:
                # PA-weighted blend for stats MLE predicts
                mle_val = mle_rates[stat]
                blended[stat] = (delegate_pa * delegate_val + mle_pa * mle_val) / total_pa
            else:
                # Use delegate for stats MLE doesn't predict
                blended[stat] = delegate_val

        return blended
