"""Training data collection for MLE model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fantasy_baseball_manager.minors.types import (
    MinorLeagueBatterSeasonStats,
    MinorLeagueLevel,
)

if TYPE_CHECKING:
    from fantasy_baseball_manager.marcel.data_source import StatsDataSource
    from fantasy_baseball_manager.marcel.models import BattingSeasonStats
    from fantasy_baseball_manager.minors.data_source import MinorLeagueDataSource
    from fantasy_baseball_manager.minors.features import MLEBatterFeatureExtractor
    from fantasy_baseball_manager.minors.types import MiLBStatcastStats
    from fantasy_baseball_manager.player_id.mapper import PlayerIdMapper

logger = logging.getLogger(__name__)


# Target stats to predict (per PA rates)
BATTER_TARGET_STATS = ("hr", "so", "bb", "singles", "doubles", "triples", "sb")


@dataclass
class AggregatedMiLBStats:
    """Aggregated stats across multiple levels for a single player-season.

    When a player plays at multiple levels in a season (e.g., 300 PA at AA + 200 PA at AAA),
    this class holds PA-weighted aggregate statistics.
    """

    player_id: str
    name: str
    season: int
    age: int
    total_pa: int
    highest_level: MinorLeagueLevel
    pct_at_aaa: float
    pct_at_aa: float
    pct_at_high_a: float
    pct_at_single_a: float

    # PA-weighted rates
    hr_rate: float
    so_rate: float
    bb_rate: float
    hit_rate: float
    singles_rate: float
    doubles_rate: float
    triples_rate: float
    sb_rate: float
    iso: float
    avg: float
    obp: float
    slg: float

    @classmethod
    def from_seasons(
        cls, seasons: list[MinorLeagueBatterSeasonStats]
    ) -> AggregatedMiLBStats | None:
        """Create aggregated stats from multiple level seasons.

        Args:
            seasons: List of season stats (can be from multiple levels).

        Returns:
            Aggregated stats, or None if no valid data.
        """
        if not seasons:
            return None

        total_pa = sum(s.pa for s in seasons)
        if total_pa == 0:
            return None

        # Use first season for player info (should be same across levels)
        first = seasons[0]

        # Find highest level (lowest sport_id = highest level)
        highest_level = min(seasons, key=lambda s: s.level.value).level

        # Calculate level distribution
        def pct_at_level(level: MinorLeagueLevel) -> float:
            pa_at_level = sum(s.pa for s in seasons if s.level == level)
            return pa_at_level / total_pa if total_pa > 0 else 0.0

        # PA-weighted rate calculations
        def weighted_rate(numerator_attr: str) -> float:
            total = sum(getattr(s, numerator_attr, 0) for s in seasons)
            return total / total_pa if total_pa > 0 else 0.0

        # PA-weighted averages for rates
        def weighted_avg(attr: str) -> float:
            return sum(getattr(s, attr, 0.0) * s.pa for s in seasons) / total_pa

        # Calculate ISO from weighted SLG - weighted AVG
        weighted_slg = weighted_avg("slg")
        weighted_avg_val = weighted_avg("avg")

        # SB rate: SB / (H + BB + HBP) - times on base
        total_on_base = sum(s.h + s.bb + s.hbp for s in seasons)
        total_sb = sum(s.sb for s in seasons)
        sb_rate = total_sb / total_on_base if total_on_base > 0 else 0.0

        return cls(
            player_id=first.player_id,
            name=first.name,
            season=first.season,
            age=first.age,
            total_pa=total_pa,
            highest_level=highest_level,
            pct_at_aaa=pct_at_level(MinorLeagueLevel.AAA),
            pct_at_aa=pct_at_level(MinorLeagueLevel.AA),
            pct_at_high_a=pct_at_level(MinorLeagueLevel.HIGH_A),
            pct_at_single_a=pct_at_level(MinorLeagueLevel.SINGLE_A),
            hr_rate=weighted_rate("hr"),
            so_rate=weighted_rate("so"),
            bb_rate=weighted_rate("bb"),
            hit_rate=weighted_rate("h"),
            singles_rate=weighted_rate("singles"),
            doubles_rate=weighted_rate("doubles"),
            triples_rate=weighted_rate("triples"),
            sb_rate=sb_rate,
            iso=weighted_slg - weighted_avg_val,
            avg=weighted_avg_val,
            obp=weighted_avg("obp"),
            slg=weighted_slg,
        )


@dataclass
class MLETrainingSample:
    """A single training sample for the MLE model."""

    player_id: str
    name: str
    milb_season: int
    mlb_season: int
    features: np.ndarray
    targets: dict[str, float]
    mlb_pa: int
    aggregated_stats: AggregatedMiLBStats | None = None


@dataclass
class MLETrainingDataCollector:
    """Collects (MiLB features, MLB outcomes) pairs for MLE model training.

    Training data alignment:
    - Features come from MiLB season Y-1 (or Y if player promoted mid-season)
    - Targets come from MLB season Y (or Y+1 if debut was late in Y)

    Inclusion criteria:
    - Player had ≥min_milb_pa at AAA or AA in year Y-1 or Y
    - Player debuted or had <max_prior_mlb_pa entering year Y
    - Player accumulated ≥min_mlb_pa in year Y or Y+1

    Exclusion criteria:
    - Players with prior MLB experience (>max_prior_mlb_pa before MiLB season)
    - September call-ups with too few PA
    """

    milb_source: MinorLeagueDataSource
    mlb_source: StatsDataSource
    min_milb_pa: int = 200
    min_mlb_pa: int = 100
    max_prior_mlb_pa: int = 200
    feature_extractor: MLEBatterFeatureExtractor | None = None
    statcast_lookup: dict[tuple[str, int], MiLBStatcastStats] | None = None
    id_mapper: PlayerIdMapper | None = None

    # Cache for feature extractor (created lazily if not provided)
    _cached_extractor: MLEBatterFeatureExtractor | None = field(
        default=None, init=False, repr=False
    )

    def collect(
        self,
        target_years: tuple[int, ...],
        include_aggregated_stats: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, list[str], list[AggregatedMiLBStats] | None]:
        """Collect training data for MLE model.

        Args:
            target_years: MLB seasons to use as targets. MiLB features come from
                         year-1 for each target year.
            include_aggregated_stats: If True, also return aggregated MiLB stats
                         for baseline comparisons.

        Returns:
            Tuple of:
            - features: (N, n_features) feature matrix
            - targets: dict mapping stat name to (N,) array of MLB rates
            - sample_weights: (N,) array of MLB PA for weighting
            - feature_names: list of feature names
            - aggregated_stats (optional): list of AggregatedMiLBStats if requested
        """
        samples = self._collect_samples(target_years)
        feature_names = self.feature_names()

        if not samples:
            logger.warning("No qualifying samples found for years %s", target_years)
            return (
                np.array([]).reshape(0, len(feature_names)),
                {stat: np.array([]) for stat in BATTER_TARGET_STATS},
                np.array([]),
                feature_names,
                [] if include_aggregated_stats else None,
            )

        # Convert samples to arrays
        features = np.array([s.features for s in samples])
        targets = {
            stat: np.array([s.targets[stat] for s in samples])
            for stat in BATTER_TARGET_STATS
        }
        weights = np.array([s.mlb_pa for s in samples], dtype=np.float32)

        logger.info(
            "Collected %d training samples from years %s",
            len(samples),
            target_years,
        )

        aggregated_stats: list[AggregatedMiLBStats] | None = None
        if include_aggregated_stats:
            aggregated_stats = [s.aggregated_stats for s in samples if s.aggregated_stats is not None]

        return features, targets, weights, feature_names, aggregated_stats

    def _collect_samples(
        self, target_years: tuple[int, ...]
    ) -> list[MLETrainingSample]:
        """Collect training samples for given target years."""
        samples: list[MLETrainingSample] = []

        for target_year in target_years:
            year_samples = self._collect_year_samples(target_year)
            samples.extend(year_samples)
            logger.info(
                "Year %d: %d qualifying samples", target_year, len(year_samples)
            )

        return samples

    def _translate_mlbam_to_fangraphs(self, mlbam_id: str) -> str | None:
        """Translate MLBAM ID to FanGraphs ID using the mapper.

        If no mapper is configured, returns the original ID (assumes same ID system).
        """
        if self.id_mapper is None:
            return mlbam_id
        return self.id_mapper.mlbam_to_fangraphs(mlbam_id)

    def _collect_year_samples(self, target_year: int) -> list[MLETrainingSample]:
        """Collect samples for a single target year.

        For each qualifying player:
        1. Get MiLB stats from year-1 (or year if promoted mid-season)
        2. Aggregate across levels
        3. Get MLB stats from target year (or year+1)
        4. Extract features and targets
        """
        samples: list[MLETrainingSample] = []
        milb_year = target_year - 1

        # Get all MiLB players from prior year with sufficient PA at upper levels
        milb_batters = self.milb_source.batting_stats_all_levels(milb_year)
        milb_by_player = self._group_by_player(milb_batters)

        # Get MLB stats for target year and year+1 (for late-season call-ups)
        mlb_target = self.mlb_source.batting_stats(target_year)
        mlb_target_next = self.mlb_source.batting_stats(target_year + 1)

        mlb_lookup = {s.player_id: s for s in mlb_target}
        mlb_next_lookup = {s.player_id: s for s in mlb_target_next}

        # Also need to check prior MLB experience
        # Players with >max_prior_mlb_pa before the MiLB season are excluded
        mlb_prior = self.mlb_source.batting_stats(milb_year)
        mlb_prior_lookup = {s.player_id: s for s in mlb_prior}

        for player_id, milb_seasons in milb_by_player.items():
            # Aggregate MiLB stats across levels
            aggregated = AggregatedMiLBStats.from_seasons(milb_seasons)
            if aggregated is None:
                continue

            # Check MiLB PA threshold
            if aggregated.total_pa < self.min_milb_pa:
                continue

            # Check that player was at upper levels (AAA or AA)
            if aggregated.highest_level not in (
                MinorLeagueLevel.AAA,
                MinorLeagueLevel.AA,
            ):
                continue

            # Translate MLBAM ID to FanGraphs ID for MLB lookups
            fg_id = self._translate_mlbam_to_fangraphs(player_id)
            if fg_id is None:
                # No mapping found, skip this player
                continue

            # Check for prior MLB experience (exclude veterans)
            prior_mlb = mlb_prior_lookup.get(fg_id)
            if prior_mlb and prior_mlb.pa > self.max_prior_mlb_pa:
                continue

            # Get MLB outcome from target year or year+1
            mlb_stats, mlb_season = self._get_mlb_outcome(
                fg_id, mlb_lookup, mlb_next_lookup, target_year
            )
            if mlb_stats is None:
                continue

            # Check MLB PA threshold
            if mlb_stats.pa < self.min_mlb_pa:
                continue

            # Extract features and targets
            features = self._extract_features(aggregated)
            targets = self._extract_targets(mlb_stats)

            samples.append(
                MLETrainingSample(
                    player_id=player_id,  # Keep MLBAM ID for consistency
                    name=aggregated.name,
                    milb_season=milb_year,
                    mlb_season=mlb_season,
                    features=features,
                    targets=targets,
                    mlb_pa=mlb_stats.pa,
                    aggregated_stats=aggregated,
                )
            )

        return samples

    def _group_by_player(
        self, stats: list[MinorLeagueBatterSeasonStats]
    ) -> dict[str, list[MinorLeagueBatterSeasonStats]]:
        """Group season stats by player ID."""
        by_player: dict[str, list[MinorLeagueBatterSeasonStats]] = {}
        for s in stats:
            if s.player_id not in by_player:
                by_player[s.player_id] = []
            by_player[s.player_id].append(s)
        return by_player

    def _get_mlb_outcome(
        self,
        player_id: str,
        mlb_lookup: dict[str, BattingSeasonStats],
        mlb_next_lookup: dict[str, BattingSeasonStats],
        target_year: int,
    ) -> tuple[BattingSeasonStats | None, int]:
        """Get MLB outcome stats for a player.

        Checks target year first, then year+1 for late-season call-ups.
        Returns (stats, year_used) or (None, 0) if not found.
        """
        # Try target year first
        mlb_stats = mlb_lookup.get(player_id)
        if mlb_stats is not None and mlb_stats.pa >= self.min_mlb_pa:
            return mlb_stats, target_year

        # Try year+1 if target year didn't have enough PA
        mlb_next = mlb_next_lookup.get(player_id)
        if mlb_next is not None and mlb_next.pa >= self.min_mlb_pa:
            return mlb_next, target_year + 1

        # If target year had some PA but not enough, still use it
        # (better than no data for evaluation purposes)
        if mlb_stats is not None:
            return mlb_stats, target_year

        return None, 0

    def _get_feature_extractor(self) -> MLEBatterFeatureExtractor:
        """Get or create the feature extractor."""
        if self.feature_extractor is not None:
            return self.feature_extractor

        if self._cached_extractor is None:
            # Import here to avoid circular imports
            from fantasy_baseball_manager.minors.features import MLEBatterFeatureExtractor

            # Use min_pa=1 since we filter by min_milb_pa in the collector
            object.__setattr__(self, "_cached_extractor", MLEBatterFeatureExtractor(min_pa=1))

        return self._cached_extractor  # type: ignore[return-value]

    def _extract_features(self, stats: AggregatedMiLBStats) -> np.ndarray:
        """Extract feature vector from aggregated MiLB stats."""
        extractor = self._get_feature_extractor()

        # Look up Statcast data if available
        statcast = None
        if self.statcast_lookup is not None:
            statcast = self.statcast_lookup.get((stats.player_id, stats.season))

        # Extract features using the extractor
        features = extractor.extract(stats, statcast)
        if features is None:
            # Should not happen since we set min_pa=1 and filter earlier
            raise ValueError(f"Feature extraction failed for {stats.player_id}")

        return features

    def _extract_targets(self, mlb_stats: BattingSeasonStats) -> dict[str, float]:
        """Extract target rates from MLB stats."""
        pa = mlb_stats.pa
        if pa == 0:
            return {stat: 0.0 for stat in BATTER_TARGET_STATS}

        return {
            "hr": mlb_stats.hr / pa,
            "so": mlb_stats.so / pa,
            "bb": mlb_stats.bb / pa,
            "singles": mlb_stats.singles / pa,
            "doubles": mlb_stats.doubles / pa,
            "triples": mlb_stats.triples / pa,
            "sb": mlb_stats.sb / pa,
        }

    def feature_names(self) -> list[str]:
        """Get list of feature names."""
        return self._get_feature_extractor().feature_names()
