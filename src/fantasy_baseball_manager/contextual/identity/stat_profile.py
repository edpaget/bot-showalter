"""Player stat profiles for identity signal.

Multi-horizon rate profiles per player, used as foundation for
player identity features in the contextual model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fantasy_baseball_manager.contextual.training.config import (
    BATTER_TARGET_STATS,
    PITCHER_TARGET_STATS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fantasy_baseball_manager.marcel.models import BattingSeasonStats, PitchingSeasonStats

# Maps target stat names to attribute names on BattingSeasonStats/PitchingSeasonStats
BATTER_STAT_ATTR_MAP: dict[str, str] = {
    "hr": "hr",
    "so": "so",
    "bb": "bb",
    "h": "h",
    "2b": "doubles",
    "3b": "triples",
}

PITCHER_STAT_ATTR_MAP: dict[str, str] = {
    "so": "so",
    "h": "h",
    "bb": "bb",
    "hr": "hr",
}


def _compute_raw_rates(
    seasons: Sequence[BattingSeasonStats | PitchingSeasonStats],
    player_type: str,
) -> tuple[dict[str, float], float] | None:
    """Compute raw rates from a list of season stats.

    Returns (rates_dict, total_opportunities) or None if seasons is empty.
    Batters use PA as denominator, pitchers use IP*3 (outs).
    """
    if not seasons:
        return None

    if player_type == "batter":
        attr_map = BATTER_STAT_ATTR_MAP
        stat_names = BATTER_TARGET_STATS
    else:
        attr_map = PITCHER_STAT_ATTR_MAP
        stat_names = PITCHER_TARGET_STATS

    # Aggregate counts across seasons
    total_counts: dict[str, int] = {stat: 0 for stat in stat_names}
    total_opps = 0.0

    for season in seasons:
        if player_type == "batter":
            total_opps += season.pa  # type: ignore[union-attr]
        else:
            total_opps += season.ip * 3  # type: ignore[union-attr]

        for stat in stat_names:
            attr = attr_map[stat]
            total_counts[stat] += getattr(season, attr)

    # Compute rates
    rates: dict[str, float] = {}
    for stat in stat_names:
        if total_opps == 0:
            rates[stat] = 0.0
        else:
            rates[stat] = total_counts[stat] / total_opps

    return rates, total_opps


@dataclass(frozen=True)
class PlayerStatProfile:
    """Multi-horizon stat profile for a single player.

    Rate keys match target stat names: hr, so, bb, h, 2b, 3b (batters),
    so, h, bb, hr (pitchers).
    """

    player_id: str
    name: str
    year: int  # "as of" year (entering this season)
    player_type: str  # "batter" | "pitcher"
    age: int  # projected to the as-of year
    handedness: str | None  # None in Phase 1

    rates_career: dict[str, float]
    rates_3yr: dict[str, float] | None
    rates_1yr: dict[str, float] | None
    rates_30d: dict[str, float] | None  # None in Phase 1

    opportunities_career: float
    opportunities_3yr: float | None
    opportunities_1yr: float | None

    def to_feature_vector(self) -> np.ndarray:
        """Convert profile to flat feature array.

        Layout: [career_rates..., 3yr_rates..., 1yr_rates..., age]
        Missing horizons use fallback chain: 1yr -> 3yr -> career.
        """
        stat_names = BATTER_TARGET_STATS if self.player_type == "batter" else PITCHER_TARGET_STATS

        # Fallback chain: 3yr -> career, 1yr -> 3yr -> career
        rates_3yr = self.rates_3yr if self.rates_3yr is not None else self.rates_career
        rates_1yr = self.rates_1yr if self.rates_1yr is not None else rates_3yr

        features: list[float] = []
        for rates in (self.rates_career, rates_3yr, rates_1yr):
            for stat in stat_names:
                features.append(rates[stat])
        features.append(float(self.age))

        return np.array(features, dtype=np.float64)

    @staticmethod
    def feature_names(player_type: str) -> list[str]:
        """Return ordered feature names matching to_feature_vector() output."""
        stat_names = BATTER_TARGET_STATS if player_type == "batter" else PITCHER_TARGET_STATS

        names: list[str] = []
        for prefix in ("career", "3yr", "1yr"):
            for stat in stat_names:
                names.append(f"{prefix}_{stat}")
        names.append("age")
        return names


class PlayerStatProfileBuilder:
    """Builds PlayerStatProfile instances from season stats."""

    def build_profile(
        self,
        player_id: str,
        name: str,
        seasons: Mapping[int, BattingSeasonStats | PitchingSeasonStats],
        year: int,
        player_type: str,
    ) -> PlayerStatProfile:
        """Build a single player's stat profile.

        Args:
            player_id: Player identifier.
            name: Player name.
            seasons: Mapping of year -> season stats for this player.
            year: The "as of" year (entering this season).
            player_type: "batter" or "pitcher".

        Returns:
            A PlayerStatProfile with computed rate horizons.
        """
        all_seasons = list(seasons.values())

        # Career: all available seasons
        career_result = _compute_raw_rates(all_seasons, player_type)
        assert career_result is not None, "build_profile called with empty seasons"
        rates_career, opps_career = career_result

        # 3yr: seasons in [year-3, year-1], require >=2 present
        three_yr_seasons = [s for y, s in seasons.items() if year - 3 <= y <= year - 1]
        rates_3yr: dict[str, float] | None = None
        opps_3yr: float | None = None
        if len(three_yr_seasons) >= 2:
            result_3yr = _compute_raw_rates(three_yr_seasons, player_type)
            if result_3yr is not None:
                rates_3yr, opps_3yr = result_3yr

        # 1yr: year-1 only
        rates_1yr: dict[str, float] | None = None
        opps_1yr: float | None = None
        if year - 1 in seasons:
            one_yr_seasons = [seasons[year - 1]]
            result_1yr = _compute_raw_rates(one_yr_seasons, player_type)
            if result_1yr is not None:
                rates_1yr, opps_1yr = result_1yr

        # Age projection: most recent season age + gap to as-of year
        most_recent_year = max(seasons.keys())
        most_recent_age = seasons[most_recent_year].age
        projected_age = most_recent_age + (year - most_recent_year)

        return PlayerStatProfile(
            player_id=player_id,
            name=name,
            year=year,
            player_type=player_type,
            age=projected_age,
            handedness=None,
            rates_career=rates_career,
            rates_3yr=rates_3yr,
            rates_1yr=rates_1yr,
            rates_30d=None,
            opportunities_career=opps_career,
            opportunities_3yr=opps_3yr,
            opportunities_1yr=opps_1yr,
        )

    def build_all_profiles(
        self,
        batting_source: object,
        pitching_source: object,
        year: int,
        history_years: list[int] | None = None,
        min_opportunities: float = 50.0,
    ) -> list[PlayerStatProfile]:
        """Build profiles for all players from data sources.

        Uses new_context(year=y) for each history year to fetch stats,
        groups by player_id, and builds profiles.

        Args:
            batting_source: DataSource[BattingSeasonStats].
            pitching_source: DataSource[PitchingSeasonStats].
            year: The "as of" year.
            history_years: Years to fetch. Defaults to [year-5, ..., year-1].
            min_opportunities: Minimum career opportunities to include.

        Returns:
            List of PlayerStatProfile instances.
        """
        from fantasy_baseball_manager.context import new_context
        from fantasy_baseball_manager.data.protocol import ALL_PLAYERS

        if history_years is None:
            history_years = list(range(year - 5, year))

        # Collect batting stats grouped by player_id
        batter_seasons: dict[str, dict[int, BattingSeasonStats]] = {}
        for hist_year in history_years:
            with new_context(year=hist_year):
                result = batting_source(ALL_PLAYERS)  # type: ignore[operator]
                if result.is_ok():
                    for stat in result.unwrap():
                        pid = stat.player_id
                        if pid not in batter_seasons:
                            batter_seasons[pid] = {}
                        batter_seasons[pid][stat.year] = stat

        # Collect pitching stats grouped by player_id
        pitcher_seasons: dict[str, dict[int, PitchingSeasonStats]] = {}
        for hist_year in history_years:
            with new_context(year=hist_year):
                result = pitching_source(ALL_PLAYERS)  # type: ignore[operator]
                if result.is_ok():
                    for stat in result.unwrap():
                        pid = stat.player_id
                        if pid not in pitcher_seasons:
                            pitcher_seasons[pid] = {}
                        pitcher_seasons[pid][stat.year] = stat

        profiles: list[PlayerStatProfile] = []

        # Build batter profiles
        for pid, seasons_dict in batter_seasons.items():
            any_season = next(iter(seasons_dict.values()))
            profile = self.build_profile(
                player_id=pid,
                name=any_season.name,
                seasons=seasons_dict,
                year=year,
                player_type="batter",
            )
            if profile.opportunities_career >= min_opportunities:
                profiles.append(profile)

        # Build pitcher profiles
        for pid, seasons_dict in pitcher_seasons.items():
            any_season = next(iter(seasons_dict.values()))
            profile = self.build_profile(
                player_id=pid,
                name=any_season.name,
                seasons=seasons_dict,
                year=year,
                player_type="pitcher",
            )
            if profile.opportunities_career >= min_opportunities:
                profiles.append(profile)

        return profiles
