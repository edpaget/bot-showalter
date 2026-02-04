"""Feature extraction for ML models combining Marcel rates with Statcast data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fantasy_baseball_manager.pipeline.batted_ball_data import PitcherBattedBallStats
    from fantasy_baseball_manager.pipeline.skill_data import BatterSkillStats
    from fantasy_baseball_manager.pipeline.statcast_data import (
        StatcastBatterStats,
        StatcastPitcherStats,
    )
    from fantasy_baseball_manager.pipeline.types import PlayerRates

# League average swing decision metrics (2023 values)
LEAGUE_AVG_CHASE_RATE = 0.30  # O-Swing%
LEAGUE_AVG_WHIFF_RATE = 0.105  # SwStr%


@dataclass(frozen=True)
class BatterFeatureExtractor:
    """Extracts feature vectors for batters from Marcel rates and Statcast data.

    Features include:
    - Marcel rates: hr, so, bb, singles, doubles, triples, sb
    - Statcast: xba, xslg, xwoba, barrel_rate, hard_hit_rate
    - Swing decision: chase_rate, whiff_rate, derived discipline metrics
    - Derived: age, age², marcel_iso, xba_minus_marcel_avg, barrel_vs_hr_ratio
    - Context: opportunities (PA)
    """

    min_pa: int = 100

    _feature_names: tuple[str, ...] = field(
        default=(
            # Marcel rates
            "marcel_hr",
            "marcel_so",
            "marcel_bb",
            "marcel_singles",
            "marcel_doubles",
            "marcel_triples",
            "marcel_sb",
            # Statcast
            "xba",
            "xslg",
            "xwoba",
            "barrel_rate",
            "hard_hit_rate",
            # Swing decision (from skill data)
            "chase_rate",
            "whiff_rate",
            "chase_minus_league_avg",
            "whiff_minus_league_avg",
            "chase_x_whiff",
            "discipline_score",
            "has_skill_data",
            # Derived
            "age",
            "age_squared",
            "marcel_iso",
            "xba_minus_marcel_avg",
            "barrel_vs_hr_ratio",
            # Context
            "opportunities",
        ),
        repr=False,
    )

    def feature_names(self) -> list[str]:
        """Return the list of feature names in order."""
        return list(self._feature_names)

    def extract(
        self,
        player: PlayerRates,
        statcast: StatcastBatterStats,
        skill_data: BatterSkillStats | None = None,
    ) -> np.ndarray | None:
        """Extract features for a batter.

        Args:
            player: Player rates from Marcel projection.
            statcast: Statcast expected stats (xBA, xSLG, etc.).
            skill_data: Optional swing decision metrics (chase_rate, whiff_rate).
                        Uses league averages if not provided.

        Returns None if player lacks required data or below minimum PA.
        """
        if statcast.pa < self.min_pa:
            return None

        rates = player.rates
        required_rates = ("hr", "so", "bb", "singles", "doubles", "triples")
        if not all(r in rates for r in required_rates):
            return None

        # Marcel rates
        hr = rates["hr"]
        so = rates["so"]
        bb = rates["bb"]
        singles = rates["singles"]
        doubles = rates["doubles"]
        triples = rates["triples"]
        sb = rates.get("sb", 0.0)

        # Swing decision metrics
        if skill_data is not None:
            chase_rate = skill_data.chase_rate
            whiff_rate = skill_data.whiff_rate
            has_skill_data = 1.0
        else:
            chase_rate = LEAGUE_AVG_CHASE_RATE
            whiff_rate = LEAGUE_AVG_WHIFF_RATE
            has_skill_data = 0.0

        chase_minus_league_avg = chase_rate - LEAGUE_AVG_CHASE_RATE
        whiff_minus_league_avg = whiff_rate - LEAGUE_AVG_WHIFF_RATE
        chase_x_whiff = chase_rate * whiff_rate
        discipline_score = (1.0 - chase_rate) * (1.0 - whiff_rate)

        # Derived from Marcel
        marcel_iso = hr + (doubles * 0.333) + (triples * 0.666)
        marcel_avg = singles + doubles + triples + hr  # Simplified BA proxy

        # Derived from Statcast
        xba_minus_marcel_avg = statcast.xba - marcel_avg
        barrel_vs_hr_ratio = statcast.barrel_rate / hr if hr > 0.001 else 0.0

        features = np.array(
            [
                hr,
                so,
                bb,
                singles,
                doubles,
                triples,
                sb,
                statcast.xba,
                statcast.xslg,
                statcast.xwoba,
                statcast.barrel_rate,
                statcast.hard_hit_rate,
                chase_rate,
                whiff_rate,
                chase_minus_league_avg,
                whiff_minus_league_avg,
                chase_x_whiff,
                discipline_score,
                has_skill_data,
                player.age,
                player.age**2,
                marcel_iso,
                xba_minus_marcel_avg,
                barrel_vs_hr_ratio,
                player.opportunities,
            ],
            dtype=np.float64,
        )

        return features


@dataclass(frozen=True)
class PitcherFeatureExtractor:
    """Extracts feature vectors for pitchers from Marcel rates, Statcast, and batted ball data.

    Features include:
    - Marcel rates: h, er, so, bb, hr
    - Statcast: xba_against, xslg_against, xwoba_against, xera, barrel_rate_against, hard_hit_rate_against
    - Batted ball: gb_pct, fb_pct, ld_pct, iffb_pct
    - Derived: age, age², xera_minus_marcel_era, gb_fb_ratio
    - Context: opportunities (outs), is_starter
    """

    min_pa: int = 100

    _feature_names: tuple[str, ...] = field(
        default=(
            # Marcel rates
            "marcel_h",
            "marcel_er",
            "marcel_so",
            "marcel_bb",
            "marcel_hr",
            # Statcast
            "xba_against",
            "xslg_against",
            "xwoba_against",
            "xera",
            "barrel_rate_against",
            "hard_hit_rate_against",
            # Batted ball
            "gb_pct",
            "fb_pct",
            "ld_pct",
            "iffb_pct",
            # Derived
            "age",
            "age_squared",
            "xera_minus_marcel_era",
            "gb_fb_ratio",
            # Context
            "opportunities",
            "is_starter",
        ),
        repr=False,
    )

    def feature_names(self) -> list[str]:
        """Return the list of feature names in order."""
        return list(self._feature_names)

    def extract(
        self,
        player: PlayerRates,
        statcast: StatcastPitcherStats,
        batted_ball: PitcherBattedBallStats | None,
    ) -> np.ndarray | None:
        """Extract features for a pitcher.

        Returns None if player lacks required data or below minimum PA against.
        """
        if statcast.pa < self.min_pa:
            return None

        rates = player.rates
        required_rates = ("h", "er", "so", "bb", "hr")
        if not all(r in rates for r in required_rates):
            return None

        # Marcel rates
        h = rates["h"]
        er = rates["er"]
        so = rates["so"]
        bb = rates["bb"]
        hr = rates["hr"]

        # Derived: Marcel ERA proxy (er rate * 9 / outs * 27 = er * 3)
        marcel_era_proxy = er * 9.0 if player.opportunities > 0 else 4.50
        xera_minus_marcel = statcast.xera - marcel_era_proxy

        # Batted ball features (use defaults if not available)
        if batted_ball is not None:
            gb_pct = batted_ball.gb_pct
            fb_pct = batted_ball.fb_pct
            ld_pct = batted_ball.ld_pct
            iffb_pct = batted_ball.iffb_pct
        else:
            # League average defaults
            gb_pct = 0.43
            fb_pct = 0.35
            ld_pct = 0.20
            iffb_pct = 0.10

        gb_fb_ratio = gb_pct / fb_pct if fb_pct > 0.01 else 1.0

        # Context
        is_starter = 1.0 if player.metadata.get("is_starter", False) else 0.0

        features = np.array(
            [
                h,
                er,
                so,
                bb,
                hr,
                statcast.xba,
                statcast.xslg,
                statcast.xwoba,
                statcast.xera,
                statcast.barrel_rate,
                statcast.hard_hit_rate,
                gb_pct,
                fb_pct,
                ld_pct,
                iffb_pct,
                player.age,
                player.age**2,
                xera_minus_marcel,
                gb_fb_ratio,
                player.opportunities,
                is_starter,
            ],
            dtype=np.float64,
        )

        return features
