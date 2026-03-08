from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain import ExpectedGamesLost
from fantasy_baseball_manager.services.il_stint_days import compute_stint_days

if TYPE_CHECKING:
    from fantasy_baseball_manager.domain import InjuryProfile

_BASE_RATE_DAYS: float = 12.0
_FULL_CREDIBILITY_SEASONS: int = 6
_RECURRENCE_BOOST_PER_LOCATION: float = 0.15
_MAX_RECURRENCE_MULTIPLIER: float = 1.5
_P_FULL_DECAY: float = 40.0
_CHRONIC_STINT_THRESHOLD: int = 4


def _recency_weight(seasons_ago: int) -> int:
    """Weight for a season: last season 3x, two ago 2x, older 1x."""
    if seasons_ago == 0:
        return 3
    if seasons_ago == 1:
        return 2
    return 1


def estimate_games_lost(profile: InjuryProfile, projection_season: int) -> ExpectedGamesLost:
    """Estimate expected days lost next season from an injury profile.

    Algorithm:
    1. Recency-weighted average of days per season from all stints.
    2. Base-rate regression toward population baseline.
    3. Recurrence multiplier for repeated injury locations.
    4. Chronic-injury floor for players with 4+ stints.
    5. p_full_season = exp(-expected_days / 40).
    6. Confidence from seasons tracked.
    """
    # Step 1: Recency-weighted average using all stints
    if profile.all_stints:
        days_by_season: dict[int, int] = defaultdict(int)
        for stint in profile.all_stints:
            days_by_season[stint.season] += compute_stint_days(stint)

        total_weighted = 0.0
        total_weight = 0
        for season, days in days_by_season.items():
            seasons_ago = projection_season - 1 - season
            w = _recency_weight(seasons_ago)
            total_weighted += days * w
            total_weight += w

        weighted_avg = total_weighted / total_weight if total_weight > 0 else 0.0
    else:
        weighted_avg = 0.0

    # Step 2: Base-rate regression
    credibility = min(1.0, profile.seasons_tracked / _FULL_CREDIBILITY_SEASONS)
    regressed = credibility * weighted_avg + (1 - credibility) * _BASE_RATE_DAYS

    # Step 3: Recurrence multiplier
    n_recurring = sum(1 for count in profile.injury_locations.values() if count >= 2)
    recurrence_multiplier = min(
        _MAX_RECURRENCE_MULTIPLIER,
        1.0 + _RECURRENCE_BOOST_PER_LOCATION * n_recurring,
    )
    expected_days = regressed * recurrence_multiplier

    # Step 4: Chronic-injury floor
    if profile.total_stints >= _CHRONIC_STINT_THRESHOLD and expected_days < _BASE_RATE_DAYS:
        expected_days = _BASE_RATE_DAYS

    # Step 5: p_full_season
    p_full = math.exp(-expected_days / _P_FULL_DECAY)

    # Step 6: Confidence
    if profile.seasons_tracked < 3:
        confidence = "low"
    elif profile.seasons_tracked < 6:
        confidence = "medium"
    else:
        confidence = "high"

    return ExpectedGamesLost(
        player_id=profile.player_id,
        expected_days_lost=round(expected_days, 1),
        p_full_season=round(p_full, 4),
        confidence=confidence,
    )
