import pytest

from fantasy_baseball_manager.models.stat_utils import (
    best_rows_per_player,
    compute_batter_rates,
    compute_pitcher_rates,
)


class TestBestRowsPerPlayer:
    def test_single_row(self) -> None:
        rows = [{"player_id": 1, "season": 2023, "proj_pa": 600}]
        result = best_rows_per_player(rows)
        assert result == {1: {"player_id": 1, "season": 2023, "proj_pa": 600}}

    def test_multiple_seasons_keeps_latest(self) -> None:
        rows = [
            {"player_id": 1, "season": 2022, "proj_pa": 550},
            {"player_id": 1, "season": 2023, "proj_pa": 600},
        ]
        result = best_rows_per_player(rows)
        assert result == {1: {"player_id": 1, "season": 2023, "proj_pa": 600}}

    def test_multiple_players(self) -> None:
        rows = [
            {"player_id": 1, "season": 2023, "proj_pa": 600},
            {"player_id": 2, "season": 2023, "proj_pa": 500},
        ]
        result = best_rows_per_player(rows)
        assert len(result) == 2
        assert result[1]["player_id"] == 1
        assert result[2]["player_id"] == 2

    def test_empty_list(self) -> None:
        result = best_rows_per_player([])
        assert result == {}


class TestComputeBatterRates:
    def test_normal_case(self) -> None:
        stats = {
            "h": 150.0,
            "doubles": 30.0,
            "triples": 3.0,
            "hr": 25.0,
            "bb": 60.0,
            "hbp": 5.0,
            "sf": 5.0,
            "ibb": 5.0,
        }
        pa = 600
        result = compute_batter_rates(stats, pa)
        ab = pa - 60 - 5 - 5  # 530
        assert result["ab"] == ab
        assert result["avg"] == pytest.approx(150.0 / ab)
        assert result["obp"] == pytest.approx((150 + 60 + 5) / (ab + 60 + 5 + 5))
        singles = 150 - 30 - 3 - 25  # 92
        slg = (singles + 2 * 30 + 3 * 3 + 4 * 25) / ab
        assert result["slg"] == pytest.approx(slg)
        assert result["ops"] == pytest.approx(result["obp"] + result["slg"])
        # woba_denom = ab + bb - ibb + sf + hbp = 530 + 60 - 5 + 5 + 5 = 595
        woba_denom = ab + 60 - 5 + 5 + 5
        expected_woba = (0.690 * 60 + 0.720 * 5 + 0.880 * singles + 1.240 * 30 + 1.560 * 3 + 2.010 * 25) / woba_denom
        assert result["woba"] == pytest.approx(expected_woba, abs=0.001)

    def test_zero_pa_returns_empty(self) -> None:
        stats = {"h": 10.0, "bb": 5.0}
        result = compute_batter_rates(stats, 0)
        assert result == {}

    def test_zero_ab_returns_empty(self) -> None:
        # pa = bb + hbp + sf → ab = 0
        stats = {"h": 0.0, "bb": 5.0, "hbp": 3.0, "sf": 2.0}
        result = compute_batter_rates(stats, 10)
        assert result == {}


class TestComputePitcherRates:
    def test_normal_case(self) -> None:
        stats = {"er": 60.0, "h": 150.0, "bb": 50.0, "so": 180.0}
        ip = 180.0
        result = compute_pitcher_rates(stats, ip)
        assert result["era"] == pytest.approx(60 * 9 / 180)
        assert result["whip"] == pytest.approx((150 + 50) / 180)
        assert result["k_per_9"] == pytest.approx(180 * 9 / 180)
        assert result["bb_per_9"] == pytest.approx(50 * 9 / 180)

    def test_zero_ip_returns_empty(self) -> None:
        stats = {"er": 0.0, "h": 0.0, "bb": 0.0, "so": 0.0}
        result = compute_pitcher_rates(stats, 0.0)
        assert result == {}
