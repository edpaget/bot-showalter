from __future__ import annotations

import math

from fantasy_baseball_manager.domain import ILStint, InjuryProfile
from fantasy_baseball_manager.services.games_lost_estimator import (
    _BASE_RATE_DAYS,
    estimate_games_lost,
)


def _profile(
    player_id: int = 1,
    seasons_tracked: int = 5,
    total_stints: int = 0,
    total_days_lost: int = 0,
    avg_days_per_season: float = 0.0,
    max_days_in_season: int = 0,
    pct_seasons_with_il: float = 0.0,
    injury_locations: dict[str, int] | None = None,
    recent_stints: list[ILStint] | None = None,
) -> InjuryProfile:
    return InjuryProfile(
        player_id=player_id,
        seasons_tracked=seasons_tracked,
        total_stints=total_stints,
        total_days_lost=total_days_lost,
        avg_days_per_season=avg_days_per_season,
        max_days_in_season=max_days_in_season,
        pct_seasons_with_il=pct_seasons_with_il,
        injury_locations=injury_locations or {},
        recent_stints=recent_stints or [],
    )


def _stint(
    player_id: int = 1,
    season: int = 2024,
    start_date: str = "2024-05-01",
    il_type: str = "10-day",
    days: int | None = 15,
    injury_location: str | None = None,
) -> ILStint:
    return ILStint(
        player_id=player_id,
        season=season,
        start_date=start_date,
        il_type=il_type,
        days=days,
        injury_location=injury_location,
    )


class TestEstimateGamesLost:
    def test_clean_player_gets_baseline(self) -> None:
        """Clean player with no stints should get regressed toward base rate."""
        profile = _profile(seasons_tracked=5)
        result = estimate_games_lost(profile, projection_season=2026)

        # credibility = 5/6, weighted_avg = 0, regressed = (5/6)*0 + (1/6)*12 = 2.0
        assert result.expected_days_lost > 0
        assert result.expected_days_lost < _BASE_RATE_DAYS
        assert result.p_full_season > 0.9
        assert result.confidence == "medium"

    def test_chronic_player_high_expected_days(self) -> None:
        """Chronically injured player should have high expected days."""
        stints = [
            _stint(season=2023, start_date="2023-04-15", days=30, injury_location="hamstring"),
            _stint(season=2024, start_date="2024-05-01", days=40, injury_location="hamstring"),
            _stint(season=2024, start_date="2024-08-01", days=20, injury_location="knee"),
        ]
        profile = _profile(
            seasons_tracked=5,
            total_stints=3,
            total_days_lost=90,
            injury_locations={"hamstring": 2, "knee": 1},
            recent_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)

        assert result.expected_days_lost > 30
        assert result.p_full_season < 0.5

    def test_single_incident_moderate_regression(self) -> None:
        """Single-incident player should be regressed toward base rate."""
        stints = [_stint(season=2024, days=60, injury_location="elbow")]
        profile = _profile(
            seasons_tracked=5,
            total_stints=1,
            total_days_lost=60,
            injury_locations={"elbow": 1},
            recent_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)

        # Should be less than 60 (regressed) but more than baseline
        assert _BASE_RATE_DAYS < result.expected_days_lost < 60

    def test_recency_weighting_recent_heavy(self) -> None:
        """Same total days but recent-heavy should produce higher estimate."""
        # Recent-heavy: 40 days in 2024 (weight 2), 10 in 2022 (weight 1)
        recent_stints = [
            _stint(season=2024, days=40),
            _stint(season=2022, days=10),
        ]
        recent_profile = _profile(
            seasons_tracked=5,
            recent_stints=recent_stints,
            injury_locations={"knee": 2},
        )

        # Old-heavy: 10 days in 2024 (weight 2), 40 in 2022 (weight 1)
        old_stints = [
            _stint(season=2024, days=10),
            _stint(season=2022, days=40),
        ]
        old_profile = _profile(
            seasons_tracked=5,
            recent_stints=old_stints,
            injury_locations={"knee": 2},
        )

        recent_result = estimate_games_lost(recent_profile, projection_season=2026)
        old_result = estimate_games_lost(old_profile, projection_season=2026)

        assert recent_result.expected_days_lost > old_result.expected_days_lost

    def test_recurrence_boost(self) -> None:
        """Repeated location injuries should boost the estimate."""
        stints = [
            _stint(season=2024, days=20, injury_location="hamstring"),
        ]

        # No recurring locations
        no_recurrence = _profile(
            seasons_tracked=5,
            recent_stints=stints,
            injury_locations={"hamstring": 1, "knee": 1},
        )

        # Recurring hamstring (2+ stints)
        with_recurrence = _profile(
            seasons_tracked=5,
            recent_stints=stints,
            injury_locations={"hamstring": 3, "knee": 1},
        )

        result_no = estimate_games_lost(no_recurrence, projection_season=2026)
        result_yes = estimate_games_lost(with_recurrence, projection_season=2026)

        assert result_yes.expected_days_lost > result_no.expected_days_lost

    def test_recurrence_multiplier_capped(self) -> None:
        """Recurrence multiplier should be capped at 1.5x."""
        stints = [_stint(season=2024, days=30)]
        # 10 recurring locations — would be 1 + 0.15*10 = 2.5, but capped at 1.5
        locations = {f"loc{i}": 3 for i in range(10)}
        profile = _profile(
            seasons_tracked=6,
            recent_stints=stints,
            injury_locations=locations,
        )
        result = estimate_games_lost(profile, projection_season=2026)

        # Full credibility, weighted_avg = 30 (weight 3, one season)
        # regressed = 30, capped multiplier = 1.5 → 45
        assert result.expected_days_lost == 45.0

    def test_confidence_low(self) -> None:
        profile = _profile(seasons_tracked=2)
        result = estimate_games_lost(profile, projection_season=2026)
        assert result.confidence == "low"

    def test_confidence_medium(self) -> None:
        profile = _profile(seasons_tracked=4)
        result = estimate_games_lost(profile, projection_season=2026)
        assert result.confidence == "medium"

    def test_confidence_high(self) -> None:
        profile = _profile(seasons_tracked=6)
        result = estimate_games_lost(profile, projection_season=2026)
        assert result.confidence == "high"

    def test_zero_seasons_tracked_pure_base_rate(self) -> None:
        """0 seasons tracked means 0 credibility → pure base rate."""
        profile = _profile(seasons_tracked=0)
        result = estimate_games_lost(profile, projection_season=2026)
        assert result.expected_days_lost == _BASE_RATE_DAYS

    def test_p_full_season_formula(self) -> None:
        """p_full_season should follow exp(-expected_days / 40)."""
        stints = [_stint(season=2024, days=20)]
        profile = _profile(
            seasons_tracked=6,
            recent_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)
        expected_p = math.exp(-result.expected_days_lost / 40)
        assert abs(result.p_full_season - round(expected_p, 4)) < 0.001

    def test_player_id_preserved(self) -> None:
        profile = _profile(player_id=42)
        result = estimate_games_lost(profile, projection_season=2026)
        assert result.player_id == 42

    def test_stint_with_no_days_uses_default_15(self) -> None:
        """Stints with days=None should default to 15 in the estimator."""
        stints = [_stint(season=2024, days=None)]
        profile = _profile(
            seasons_tracked=6,
            recent_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)
        # Full credibility, weighted_avg = 15 (one stint, weight 3)
        # regressed = 15, no recurrence → 15.0
        assert result.expected_days_lost == 15.0
