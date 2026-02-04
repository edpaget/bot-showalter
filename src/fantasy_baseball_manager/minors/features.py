"""Feature extraction for MLE model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from fantasy_baseball_manager.minors.types import MinorLeagueLevel

if TYPE_CHECKING:
    from fantasy_baseball_manager.minors.training_data import AggregatedMiLBStats
    from fantasy_baseball_manager.minors.types import MiLBStatcastStats


# Typical age for each MiLB level (used for age-for-level feature)
TYPICAL_AGE_BY_LEVEL: dict[MinorLeagueLevel, float] = {
    MinorLeagueLevel.AAA: 25.0,
    MinorLeagueLevel.AA: 23.0,
    MinorLeagueLevel.HIGH_A: 22.0,
    MinorLeagueLevel.SINGLE_A: 21.0,
    MinorLeagueLevel.ROOKIE: 19.0,
}


@dataclass(frozen=True)
class MLEBatterFeatureExtractor:
    """Extracts feature vectors for MLE model from aggregated MiLB stats.

    Features include:
    - Rate stats: hr, so, bb, hit, singles, doubles, triples, sb, iso, avg, obp, slg
    - Age features: age, age_squared, age_for_level
    - Level encoding: one-hot for highest level
    - Level distribution: percentage of PA at each level
    - Sample size: total_pa, log_pa
    - Statcast (when available): xba, xslg, xwoba, barrel_rate, hard_hit_rate, sprint_speed

    The extractor handles missing Statcast data using indicator + zero-imputation:
    when Statcast data is not available, features are set to 0 and has_statcast=0
    so the model can learn to ignore them.
    """

    min_pa: int = 50

    _feature_names: tuple[str, ...] = field(
        default=(
            # Rate stats (12 features)
            "hr_rate",
            "so_rate",
            "bb_rate",
            "hit_rate",
            "singles_rate",
            "doubles_rate",
            "triples_rate",
            "sb_rate",
            "iso",
            "avg",
            "obp",
            "slg",
            # Age features (3 features)
            "age",
            "age_squared",
            "age_for_level",
            # Level one-hot (4 features)
            "level_aaa",
            "level_aa",
            "level_high_a",
            "level_single_a",
            # Level distribution (4 features)
            "pct_at_aaa",
            "pct_at_aa",
            "pct_at_high_a",
            "pct_at_single_a",
            # Sample size (2 features)
            "total_pa",
            "log_pa",
            # Statcast features (7 features)
            "xba",
            "xslg",
            "xwoba",
            "barrel_rate",
            "hard_hit_rate",
            "sprint_speed",
            "has_statcast",
        ),
        repr=False,
    )

    def feature_names(self) -> list[str]:
        """Return the list of feature names in order."""
        return list(self._feature_names)

    def n_features(self) -> int:
        """Return the number of features."""
        return len(self._feature_names)

    def extract(
        self,
        stats: AggregatedMiLBStats,
        statcast: MiLBStatcastStats | None = None,
    ) -> np.ndarray | None:
        """Extract features for a batter.

        Args:
            stats: Aggregated MiLB stats across levels for a player-season.
            statcast: Optional Statcast data (AAA 2023+ only).

        Returns:
            Feature vector as numpy array, or None if below minimum PA threshold.
        """
        if stats.total_pa < self.min_pa:
            return None

        # Age for level: how old is this player relative to typical age at their level
        typical_age = TYPICAL_AGE_BY_LEVEL.get(stats.highest_level, 23.0)
        age_for_level = stats.age - typical_age

        # Statcast features with zero-imputation
        if statcast is not None:
            xba = statcast.xba
            xslg = statcast.xslg
            xwoba = statcast.xwoba
            barrel_rate = statcast.barrel_rate
            hard_hit_rate = statcast.hard_hit_rate
            sprint_speed = statcast.sprint_speed if statcast.sprint_speed is not None else 0.0
            has_statcast = 1.0
        else:
            xba = 0.0
            xslg = 0.0
            xwoba = 0.0
            barrel_rate = 0.0
            hard_hit_rate = 0.0
            sprint_speed = 0.0
            has_statcast = 0.0

        features = np.array(
            [
                # Rate stats
                stats.hr_rate,
                stats.so_rate,
                stats.bb_rate,
                stats.hit_rate,
                stats.singles_rate,
                stats.doubles_rate,
                stats.triples_rate,
                stats.sb_rate,
                stats.iso,
                stats.avg,
                stats.obp,
                stats.slg,
                # Age features
                float(stats.age),
                stats.age**2 / 1000.0,  # Scaled to prevent large values
                age_for_level,
                # Level one-hot
                1.0 if stats.highest_level == MinorLeagueLevel.AAA else 0.0,
                1.0 if stats.highest_level == MinorLeagueLevel.AA else 0.0,
                1.0 if stats.highest_level == MinorLeagueLevel.HIGH_A else 0.0,
                1.0 if stats.highest_level == MinorLeagueLevel.SINGLE_A else 0.0,
                # Level distribution
                stats.pct_at_aaa,
                stats.pct_at_aa,
                stats.pct_at_high_a,
                stats.pct_at_single_a,
                # Sample size
                float(stats.total_pa),
                np.log(stats.total_pa + 1),
                # Statcast
                xba,
                xslg,
                xwoba,
                barrel_rate,
                hard_hit_rate,
                sprint_speed,
                has_statcast,
            ],
            dtype=np.float32,
        )

        return features

    def extract_batch(
        self,
        stats_list: list[AggregatedMiLBStats],
        statcast_lookup: dict[str, MiLBStatcastStats] | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """Extract features for multiple batters.

        Args:
            stats_list: List of aggregated MiLB stats.
            statcast_lookup: Optional dict mapping player_id to Statcast data.

        Returns:
            Tuple of:
            - Feature matrix (N, n_features) for valid players
            - List of indices of valid players in the original list
        """
        features_list: list[np.ndarray] = []
        valid_indices: list[int] = []

        for i, stats in enumerate(stats_list):
            statcast = None
            if statcast_lookup is not None:
                statcast = statcast_lookup.get(stats.player_id)

            features = self.extract(stats, statcast)
            if features is not None:
                features_list.append(features)
                valid_indices.append(i)

        if not features_list:
            return np.array([]).reshape(0, self.n_features()), []

        return np.stack(features_list), valid_indices
