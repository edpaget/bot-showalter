from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.services.injury_discount import (
    apply_injury_discount,
    discount_projections,
)

# ---------------------------------------------------------------------------
# apply_injury_discount — batters
# ---------------------------------------------------------------------------


class TestApplyInjuryDiscountBatter:
    def test_counting_stats_scale_by_discount_factor(self) -> None:
        stats = {"pa": 600, "hr": 40.0, "r": 100.0, "rbi": 90.0, "sb": 20.0, "h": 170.0, "ab": 550.0}
        result = apply_injury_discount(stats, expected_days_lost=36.6, player_type="batter")
        factor = 1.0 - 36.6 / 183
        assert result["pa"] == 600 * factor
        assert result["hr"] == 40.0 * factor
        assert result["r"] == 100.0 * factor
        assert result["rbi"] == 90.0 * factor
        assert result["sb"] == 20.0 * factor

    def test_rate_stats_unchanged(self) -> None:
        stats = {"pa": 600, "hr": 40.0, "avg": 0.300, "obp": 0.380, "slg": 0.550}
        result = apply_injury_discount(stats, expected_days_lost=30.0, player_type="batter")
        assert result["avg"] == 0.300
        assert result["obp"] == 0.380
        assert result["slg"] == 0.550

    def test_zero_days_lost_no_change(self) -> None:
        stats = {"pa": 600, "hr": 40.0, "avg": 0.300}
        result = apply_injury_discount(stats, expected_days_lost=0.0, player_type="batter")
        assert result == stats

    def test_high_days_lost_floors_at_zero(self) -> None:
        stats = {"pa": 600, "hr": 40.0, "avg": 0.300}
        result = apply_injury_discount(stats, expected_days_lost=200.0, player_type="batter")
        assert result["pa"] == 0.0
        assert result["hr"] == 0.0
        assert result["avg"] == 0.300


# ---------------------------------------------------------------------------
# apply_injury_discount — pitchers
# ---------------------------------------------------------------------------


class TestApplyInjuryDiscountPitcher:
    def test_counting_stats_scale_by_discount_factor(self) -> None:
        stats = {"ip": 200.0, "w": 15.0, "k": 220.0, "sv": 0.0}
        result = apply_injury_discount(stats, expected_days_lost=36.6, player_type="pitcher")
        factor = 1.0 - 36.6 / 183
        assert result["ip"] == 200.0 * factor
        assert result["w"] == 15.0 * factor
        assert result["k"] == 220.0 * factor
        assert result["sv"] == 0.0 * factor

    def test_rate_stats_unchanged(self) -> None:
        stats = {"ip": 200.0, "era": 3.25, "whip": 1.10, "k9": 9.5}
        result = apply_injury_discount(stats, expected_days_lost=30.0, player_type="pitcher")
        assert result["era"] == 3.25
        assert result["whip"] == 1.10
        assert result["k9"] == 9.5

    def test_zero_days_lost_no_change(self) -> None:
        stats = {"ip": 200.0, "w": 15.0, "era": 3.25}
        result = apply_injury_discount(stats, expected_days_lost=0.0, player_type="pitcher")
        assert result == stats

    def test_high_days_lost_floors_at_zero(self) -> None:
        stats = {"ip": 200.0, "w": 15.0, "era": 3.25}
        result = apply_injury_discount(stats, expected_days_lost=200.0, player_type="pitcher")
        assert result["ip"] == 0.0
        assert result["w"] == 0.0
        assert result["era"] == 3.25


# ---------------------------------------------------------------------------
# discount_projections
# ---------------------------------------------------------------------------


def _make_projection(player_id: int, player_type: str, stat_json: dict) -> Projection:
    return Projection(
        player_id=player_id,
        season=2026,
        system="steamer",
        version="v1",
        player_type=player_type,
        stat_json=stat_json,
    )


class TestDiscountProjections:
    def test_players_in_map_get_adjusted(self) -> None:
        proj = _make_projection(1, "batter", {"pa": 600, "hr": 40.0, "avg": 0.300})
        result = discount_projections([proj], {1: 36.6})
        factor = 1.0 - 36.6 / 183
        assert result[0].stat_json["pa"] == 600 * factor
        assert result[0].stat_json["hr"] == 40.0 * factor
        assert result[0].stat_json["avg"] == 0.300

    def test_players_not_in_map_unchanged(self) -> None:
        proj = _make_projection(99, "batter", {"pa": 600, "hr": 40.0})
        result = discount_projections([proj], {1: 36.6})
        assert result[0].stat_json == {"pa": 600, "hr": 40.0}

    def test_empty_map_returns_unchanged(self) -> None:
        proj = _make_projection(1, "batter", {"pa": 600, "hr": 40.0})
        result = discount_projections([proj], {})
        assert result[0].stat_json == {"pa": 600, "hr": 40.0}

    def test_preserves_projection_metadata(self) -> None:
        proj = _make_projection(1, "batter", {"pa": 600, "hr": 40.0})
        result = discount_projections([proj], {1: 30.0})
        assert result[0].player_id == 1
        assert result[0].season == 2026
        assert result[0].system == "steamer"
        assert result[0].version == "v1"
        assert result[0].player_type == "batter"

    def test_mixed_batters_and_pitchers(self) -> None:
        batter = _make_projection(1, "batter", {"pa": 600, "hr": 40.0, "avg": 0.300})
        pitcher = _make_projection(2, "pitcher", {"ip": 200.0, "w": 15.0, "era": 3.25})
        result = discount_projections([batter, pitcher], {1: 30.0, 2: 45.0})
        # Batter counting stats discounted
        assert result[0].stat_json["pa"] < 600
        assert result[0].stat_json["avg"] == 0.300
        # Pitcher counting stats discounted
        assert result[1].stat_json["ip"] < 200.0
        assert result[1].stat_json["era"] == 3.25

    def test_returns_new_projection_objects(self) -> None:
        proj = _make_projection(1, "batter", {"pa": 600, "hr": 40.0})
        result = discount_projections([proj], {1: 30.0})
        assert result[0] is not proj
        assert result[0].stat_json is not proj.stat_json
