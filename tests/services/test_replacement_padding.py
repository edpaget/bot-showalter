from __future__ import annotations

import pytest

from fantasy_baseball_manager.domain.identity import PlayerType
from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.domain.replacement_profile import ReplacementProfile
from fantasy_baseball_manager.services.replacement_padding import (
    blend_projections,
    blend_stat_line,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_projection(
    player_id: int,
    player_type: PlayerType,
    stat_json: dict,
    *,
    season: int = 2026,
) -> Projection:
    return Projection(
        player_id=player_id,
        season=season,
        system="test",
        version="1",
        player_type=player_type,
        stat_json=stat_json,
    )


# ---------------------------------------------------------------------------
# blend_stat_line tests
# ---------------------------------------------------------------------------


class TestBlendStatLine:
    def test_counting_stats_blended(self) -> None:
        stat_json = {"hr": 30.0, "rbi": 100.0, "pa": 600.0}
        replacement = {"hr": 10.0, "rbi": 50.0, "pa": 600.0}
        # 20% missed → healthy_frac = 0.8
        result = blend_stat_line(stat_json, replacement, 183 * 0.2, "batter")
        assert result["hr"] == pytest.approx(30 * 0.8 + 10 * 0.2)  # 26.0
        assert result["rbi"] == pytest.approx(100 * 0.8 + 50 * 0.2)  # 90.0

    def test_rate_stats_blended(self) -> None:
        stat_json = {"avg": 0.300, "obp": 0.400, "pa": 600.0}
        replacement = {"avg": 0.250, "obp": 0.320}
        result = blend_stat_line(stat_json, replacement, 183 * 0.2, "batter")
        assert result["avg"] == pytest.approx(0.300 * 0.8 + 0.250 * 0.2)  # 0.290
        assert result["obp"] == pytest.approx(0.400 * 0.8 + 0.320 * 0.2)  # 0.384

    def test_volume_stats_preserved(self) -> None:
        stat_json = {"pa": 600.0, "ab": 550.0, "ip": 200.0, "g": 162, "gs": 30, "hr": 30.0}
        replacement = {"pa": 500.0, "ab": 450.0, "ip": 150.0, "g": 120, "gs": 20, "hr": 10.0}
        result = blend_stat_line(stat_json, replacement, 183 * 0.2, "batter")
        assert result["pa"] == 600.0
        assert result["ab"] == 550.0
        assert result["ip"] == 200.0
        assert result["g"] == 162
        assert result["gs"] == 30

    def test_zero_days_lost_unchanged(self) -> None:
        stat_json = {"hr": 30.0, "avg": 0.300, "pa": 600.0}
        replacement = {"hr": 10.0, "avg": 0.250, "pa": 500.0}
        result = blend_stat_line(stat_json, replacement, 0.0, "batter")
        assert result == stat_json

    def test_full_season_lost_pure_replacement(self) -> None:
        stat_json = {"hr": 30.0, "avg": 0.300, "pa": 600.0}
        replacement = {"hr": 10.0, "avg": 0.250}
        result = blend_stat_line(stat_json, replacement, 183, "batter")
        assert result["hr"] == pytest.approx(10.0)
        assert result["avg"] == pytest.approx(0.250)
        assert result["pa"] == 600.0  # volume preserved

    def test_over_183_days_same_as_183(self) -> None:
        stat_json = {"hr": 30.0, "avg": 0.300, "pa": 600.0}
        replacement = {"hr": 10.0, "avg": 0.250}
        result_183 = blend_stat_line(stat_json, replacement, 183, "batter")
        result_200 = blend_stat_line(stat_json, replacement, 200, "batter")
        assert result_200 == result_183

    def test_non_numeric_values_preserved(self) -> None:
        stat_json = {"hr": 30.0, "name": "Test Player", "tags": {"elite": True}}
        replacement = {"hr": 10.0}
        result = blend_stat_line(stat_json, replacement, 183 * 0.2, "batter")
        assert result["name"] == "Test Player"
        assert result["tags"] == {"elite": True}

    def test_missing_replacement_stat_counting(self) -> None:
        """Counting stat present in player but not replacement: scale by healthy_frac."""
        stat_json = {"hr": 30.0, "sb": 20.0, "pa": 600.0}
        replacement = {"hr": 10.0}  # no sb
        result = blend_stat_line(stat_json, replacement, 183 * 0.2, "batter")
        assert result["hr"] == pytest.approx(30 * 0.8 + 10 * 0.2)
        # sb missing from replacement → defaults to 0.0
        assert result["sb"] == pytest.approx(20 * 0.8 + 0.0 * 0.2)

    def test_missing_replacement_stat_rate(self) -> None:
        """Rate stat present in player but not replacement: preserve original."""
        stat_json = {"avg": 0.300, "woba": 0.370, "pa": 600.0}
        replacement = {"avg": 0.250}  # no woba
        result = blend_stat_line(stat_json, replacement, 183 * 0.2, "batter")
        assert result["avg"] == pytest.approx(0.300 * 0.8 + 0.250 * 0.2)
        # woba missing from replacement → uses player's own value as fallback
        assert result["woba"] == pytest.approx(0.370 * 0.8 + 0.370 * 0.2)  # == 0.370

    def test_pitcher_rate_stats(self) -> None:
        stat_json = {"era": 3.00, "whip": 1.10, "w": 15.0, "ip": 200.0}
        replacement = {"era": 4.50, "whip": 1.35, "w": 8.0}
        result = blend_stat_line(stat_json, replacement, 183 * 0.2, "pitcher")
        assert result["era"] == pytest.approx(3.00 * 0.8 + 4.50 * 0.2)
        assert result["whip"] == pytest.approx(1.10 * 0.8 + 1.35 * 0.2)
        assert result["ip"] == 200.0  # volume preserved


# ---------------------------------------------------------------------------
# blend_projections tests
# ---------------------------------------------------------------------------


class TestBlendProjections:
    def test_applies_per_player(self) -> None:
        p1 = _make_projection(1, PlayerType.BATTER, {"hr": 40.0, "avg": 0.300, "pa": 600.0})
        p2 = _make_projection(2, PlayerType.BATTER, {"hr": 20.0, "avg": 0.280, "pa": 550.0})
        profiles = {
            "OF": ReplacementProfile("OF", PlayerType.BATTER, {"hr": 10.0, "avg": 0.250}),
        }
        injury_map = {1: 183 * 0.2, 2: 183 * 0.5}  # 20%, 50%
        position_map = {1: ["OF"], 2: ["OF"]}

        result = blend_projections([p1, p2], profiles, injury_map, position_map)
        assert result[0].stat_json["hr"] == pytest.approx(40 * 0.8 + 10 * 0.2)
        assert result[1].stat_json["hr"] == pytest.approx(20 * 0.5 + 10 * 0.5)

    def test_skips_players_not_in_injury_map(self) -> None:
        p1 = _make_projection(1, PlayerType.BATTER, {"hr": 40.0, "pa": 600.0})
        p2 = _make_projection(2, PlayerType.BATTER, {"hr": 20.0, "pa": 550.0})
        profiles = {
            "OF": ReplacementProfile("OF", PlayerType.BATTER, {"hr": 10.0}),
        }
        injury_map = {1: 36.6}  # only player 1
        position_map = {1: ["OF"], 2: ["OF"]}

        result = blend_projections([p1, p2], profiles, injury_map, position_map)
        assert result[1].stat_json == p2.stat_json  # unchanged

    def test_picks_correct_position_replacement(self) -> None:
        p1 = _make_projection(1, PlayerType.BATTER, {"hr": 30.0, "pa": 600.0})
        profiles = {
            "1B": ReplacementProfile("1B", PlayerType.BATTER, {"hr": 15.0}),
            "OF": ReplacementProfile("OF", PlayerType.BATTER, {"hr": 8.0}),
        }
        injury_map = {1: 183 * 0.5}
        # First position is 1B → use 1B replacement
        position_map = {1: ["1B", "OF"]}

        result = blend_projections([p1], profiles, injury_map, position_map)
        assert result[0].stat_json["hr"] == pytest.approx(30 * 0.5 + 15 * 0.5)

    def test_falls_back_to_second_position(self) -> None:
        """If first position has no profile, use next available."""
        p1 = _make_projection(1, PlayerType.BATTER, {"hr": 30.0, "pa": 600.0})
        profiles = {
            "OF": ReplacementProfile("OF", PlayerType.BATTER, {"hr": 8.0}),
        }
        injury_map = {1: 183 * 0.5}
        position_map = {1: ["1B", "OF"]}  # 1B has no profile

        result = blend_projections([p1], profiles, injury_map, position_map)
        assert result[0].stat_json["hr"] == pytest.approx(30 * 0.5 + 8 * 0.5)

    def test_empty_injury_map(self) -> None:
        p1 = _make_projection(1, PlayerType.BATTER, {"hr": 40.0, "pa": 600.0})
        profiles = {"OF": ReplacementProfile("OF", PlayerType.BATTER, {"hr": 10.0})}

        result = blend_projections([p1], profiles, {}, {1: ["OF"]})
        assert result[0].stat_json == p1.stat_json

    def test_preserves_volume_stats(self) -> None:
        p1 = _make_projection(1, PlayerType.BATTER, {"hr": 30.0, "pa": 600.0, "ab": 550.0})
        profiles = {"OF": ReplacementProfile("OF", PlayerType.BATTER, {"hr": 10.0, "pa": 500.0, "ab": 450.0})}
        injury_map = {1: 183 * 0.3}
        position_map = {1: ["OF"]}

        result = blend_projections([p1], profiles, injury_map, position_map)
        assert result[0].stat_json["pa"] == 600.0
        assert result[0].stat_json["ab"] == 550.0
