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
    all_stints: list[ILStint] | None = None,
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
        all_stints=all_stints or [],
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
            all_stints=stints,
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
            all_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)

        # Should be less than 60 (regressed) but more than baseline
        assert _BASE_RATE_DAYS < result.expected_days_lost < 60

    def test_recency_weighting_recent_heavy(self) -> None:
        """Same total days but recent-heavy should produce higher estimate."""
        # Recent-heavy: 40 days in 2024 (weight 2), 10 in 2022 (weight 1)
        recent_stints = [
            _stint(season=2024, days=40),
            _stint(season=2022, start_date="2022-05-01", days=10),
        ]
        recent_profile = _profile(
            seasons_tracked=5,
            all_stints=recent_stints,
            injury_locations={"knee": 2},
        )

        # Old-heavy: 10 days in 2024 (weight 2), 40 in 2022 (weight 1)
        old_stints = [
            _stint(season=2024, days=10),
            _stint(season=2022, start_date="2022-05-01", days=40),
        ]
        old_profile = _profile(
            seasons_tracked=5,
            all_stints=old_stints,
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
            all_stints=stints,
            injury_locations={"hamstring": 1, "knee": 1},
        )

        # Recurring hamstring (2+ stints)
        with_recurrence = _profile(
            seasons_tracked=5,
            all_stints=stints,
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
            all_stints=stints,
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
            all_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)
        expected_p = math.exp(-result.expected_days_lost / 40)
        assert abs(result.p_full_season - round(expected_p, 4)) < 0.001

    def test_player_id_preserved(self) -> None:
        profile = _profile(player_id=42)
        result = estimate_games_lost(profile, projection_season=2026)
        assert result.player_id == 42

    def test_il_type_used_when_days_none(self) -> None:
        """Stints with days=None should use IL type default, not hardcoded 15."""
        stints = [_stint(season=2024, days=None, il_type="60-day")]
        profile = _profile(
            seasons_tracked=6,
            all_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)
        # Full credibility, weighted_avg = 60 (one 60-day stint, weight 3)
        # regressed = 60, no recurrence → 60.0
        assert result.expected_days_lost == 60.0

    def test_10_day_il_default(self) -> None:
        """10-day IL stint with days=None should default to 10."""
        stints = [_stint(season=2024, days=None, il_type="10-day")]
        profile = _profile(
            seasons_tracked=6,
            all_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)
        assert result.expected_days_lost == 10.0

    def test_multi_season_history_all_contribute(self) -> None:
        """Stints across 5 seasons should all contribute with recency weights."""
        stints = [
            _stint(season=2021, start_date="2021-05-01", days=30),
            _stint(season=2022, start_date="2022-05-01", days=40),
            _stint(season=2023, start_date="2023-05-01", days=20),
            _stint(season=2024, start_date="2024-05-01", days=50),
            _stint(season=2025, start_date="2025-05-01", days=10),
        ]
        profile = _profile(
            seasons_tracked=5,
            total_stints=5,
            all_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)

        # All seasons contribute: 2025 (w=3), 2024 (w=2), 2021-2023 (w=1 each)
        # weighted = 10*3 + 50*2 + 20*1 + 40*1 + 30*1 = 30+100+20+40+30 = 220
        # total_weight = 3+2+1+1+1 = 8
        # weighted_avg = 220/8 = 27.5
        # credibility = 5/6, regressed = (5/6)*27.5 + (1/6)*12 = 22.917 + 2 = 24.917
        assert result.expected_days_lost > 20

    def test_chronic_floor_applied(self) -> None:
        """Player with 4+ stints should get at least _BASE_RATE_DAYS."""
        # Old stints that would otherwise average to very little
        stints = [
            _stint(season=2021, start_date="2021-05-01", days=5),
            _stint(season=2021, start_date="2021-07-01", days=5),
            _stint(season=2022, start_date="2022-05-01", days=5),
            _stint(season=2022, start_date="2022-07-01", days=5),
        ]
        profile = _profile(
            seasons_tracked=5,
            total_stints=4,
            all_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)

        # Without floor: weighted_avg = (10+10)/(1+1) = 10
        # credibility = 5/6, regressed = (5/6)*10 + (1/6)*12 = 10.33
        # But with 4 stints and total_stints >= 4, floor applies if < 12
        assert result.expected_days_lost >= _BASE_RATE_DAYS

    def test_chronic_floor_not_applied_below_threshold(self) -> None:
        """Player with < 4 stints should not get the chronic floor."""
        stints = [
            _stint(season=2021, start_date="2021-05-01", days=5),
            _stint(season=2022, start_date="2022-05-01", days=5),
        ]
        profile = _profile(
            seasons_tracked=5,
            total_stints=2,
            all_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)

        # weighted_avg = (5+5)/(1+1) = 5
        # credibility = 5/6, regressed = (5/6)*5 + (1/6)*12 = 4.17 + 2 = 6.17
        assert result.expected_days_lost < _BASE_RATE_DAYS

    def test_all_stints_old_still_meaningful(self) -> None:
        """Player with stints only 3+ seasons ago should still get a meaningful estimate."""
        stints = [
            _stint(season=2021, start_date="2021-05-01", days=40),
            _stint(season=2022, start_date="2022-05-01", days=50),
        ]
        profile = _profile(
            seasons_tracked=5,
            total_stints=2,
            all_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)

        # Both get weight 1 (3+ seasons ago)
        # weighted_avg = (40+50)/2 = 45
        # credibility = 5/6, regressed = (5/6)*45 + (1/6)*12 = 37.5 + 2 = 39.5
        assert result.expected_days_lost > 30

    def test_scherzer_like_profile(self) -> None:
        """Scherzer-like profile (17 stints, 255 days, 5 seasons) → expected days >> baseline."""
        stints = [
            # 2021: 2 stints, 30 days
            _stint(season=2021, start_date="2021-05-01", days=15),
            _stint(season=2021, start_date="2021-08-01", days=15),
            # 2022: 3 stints, 45 days
            _stint(season=2022, start_date="2022-04-01", days=20),
            _stint(season=2022, start_date="2022-07-01", days=15),
            _stint(season=2022, start_date="2022-09-01", days=10),
            # 2023: 4 stints, 60 days
            _stint(season=2023, start_date="2023-04-01", days=20),
            _stint(season=2023, start_date="2023-06-01", days=15),
            _stint(season=2023, start_date="2023-08-01", days=15),
            _stint(season=2023, start_date="2023-09-01", days=10),
            # 2024: 5 stints, 80 days
            _stint(season=2024, start_date="2024-04-01", days=25),
            _stint(season=2024, start_date="2024-05-15", days=20),
            _stint(season=2024, start_date="2024-07-01", days=15),
            _stint(season=2024, start_date="2024-08-01", days=10),
            _stint(season=2024, start_date="2024-09-01", days=10),
            # 2025: 3 stints, 40 days
            _stint(season=2025, start_date="2025-04-01", days=20),
            _stint(season=2025, start_date="2025-06-01", days=10),
            _stint(season=2025, start_date="2025-08-01", days=10),
        ]
        profile = _profile(
            seasons_tracked=5,
            total_stints=17,
            total_days_lost=255,
            injury_locations={"shoulder": 8, "back": 5, "neck": 4},
            all_stints=stints,
        )
        result = estimate_games_lost(profile, projection_season=2026)

        # With 17 stints, multi-season history, and recurring locations → well above baseline
        assert result.expected_days_lost > 25
