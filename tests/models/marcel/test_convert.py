import pytest

from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.marcel.convert import (
    extract_pt_from_rows,
    projection_to_domain,
    rows_to_marcel_inputs,
    rows_to_player_seasons,
)
from fantasy_baseball_manager.models.marcel.types import MarcelInput, MarcelProjection


class TestRowsToPlayerSeasons:
    def test_single_player_three_lags(self) -> None:
        rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 32,
                "pa_1": 600,
                "pa_2": 500,
                "pa_3": 400,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": 20.0,
            },
        ]
        result = rows_to_player_seasons(rows, ["hr"], lags=3)
        assert 1 in result
        player_id, seasons, age = result[1]
        assert player_id == 1
        assert age == 32
        assert len(seasons) == 3
        # Most recent first
        assert seasons[0].pa == 600
        assert seasons[0].stats["hr"] == 30.0
        assert seasons[1].pa == 500
        assert seasons[1].stats["hr"] == 25.0
        assert seasons[2].pa == 400
        assert seasons[2].stats["hr"] == 20.0

    def test_multiple_players(self) -> None:
        rows = [
            {"player_id": 1, "season": 2023, "age": 32, "pa_1": 600, "hr_1": 30.0},
            {"player_id": 2, "season": 2023, "age": 25, "pa_1": 500, "hr_1": 20.0},
        ]
        result = rows_to_player_seasons(rows, ["hr"], lags=1)
        assert len(result) == 2
        assert 1 in result
        assert 2 in result

    def test_pitcher_rows(self) -> None:
        rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 28,
                "ip_1": 180.0,
                "ip_2": 170.0,
                "g_1": 30,
                "g_2": 28,
                "gs_1": 30,
                "gs_2": 28,
                "so_1": 200.0,
                "so_2": 180.0,
            },
        ]
        result = rows_to_player_seasons(rows, ["so"], lags=2, pitcher=True)
        player_id, seasons, age = result[1]
        assert len(seasons) == 2
        assert seasons[0].ip == 180.0
        assert seasons[0].g == 30
        assert seasons[0].gs == 30
        assert seasons[0].stats["so"] == 200.0

    def test_missing_lag_columns_produce_zero(self) -> None:
        rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 30,
                "pa_1": 600,
                "pa_2": 0,
                "hr_1": 30.0,
                "hr_2": 0.0,
            },
        ]
        result = rows_to_player_seasons(rows, ["hr"], lags=2)
        _, seasons, _ = result[1]
        assert seasons[1].pa == 0
        assert seasons[1].stats["hr"] == 0.0

    def test_uses_most_recent_row_per_player(self) -> None:
        rows = [
            {"player_id": 1, "season": 2022, "age": 31, "pa_1": 500, "hr_1": 25.0},
            {"player_id": 1, "season": 2023, "age": 32, "pa_1": 600, "hr_1": 30.0},
        ]
        result = rows_to_player_seasons(rows, ["hr"], lags=1)
        _, seasons, age = result[1]
        assert age == 32
        assert seasons[0].pa == 600


class TestRowsToMarcelInputs:
    def test_single_batter(self) -> None:
        rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 550,
                "pa_3": 500,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": 20.0,
                "hr_wavg": 0.05,
                "weighted_pt": 6200.0,
                "league_hr_rate": 0.03,
            },
        ]
        result = rows_to_marcel_inputs(rows, ["hr"], lags=3)
        assert 1 in result
        inp = result[1]
        assert isinstance(inp, MarcelInput)
        assert inp.age == 29
        assert inp.weighted_rates == {"hr": 0.05}
        assert inp.weighted_pt == 6200.0
        assert inp.league_rates == {"hr": 0.03}
        assert len(inp.seasons) == 3
        assert inp.seasons[0].pa == 600
        assert inp.seasons[1].pa == 550

    def test_multiple_categories(self) -> None:
        rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "pa_2": 500,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "h_1": 150.0,
                "h_2": 130.0,
                "hr_wavg": 0.05,
                "h_wavg": 0.27,
                "weighted_pt": 5000.0,
                "league_hr_rate": 0.03,
                "league_h_rate": 0.25,
            },
        ]
        result = rows_to_marcel_inputs(rows, ["hr", "h"], lags=2)
        inp = result[1]
        assert inp.weighted_rates == {"hr": 0.05, "h": 0.27}
        assert inp.league_rates == {"hr": 0.03, "h": 0.25}

    def test_multiple_players(self) -> None:
        rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "hr_1": 30.0,
                "hr_wavg": 0.05,
                "weighted_pt": 3000.0,
                "league_hr_rate": 0.03,
            },
            {
                "player_id": 2,
                "season": 2023,
                "age": 25,
                "pa_1": 500,
                "hr_1": 20.0,
                "hr_wavg": 0.04,
                "weighted_pt": 2500.0,
                "league_hr_rate": 0.03,
            },
        ]
        result = rows_to_marcel_inputs(rows, ["hr"], lags=1)
        assert len(result) == 2
        assert result[1].age == 29
        assert result[2].age == 25

    def test_pitcher_builds_season_lines_with_ip(self) -> None:
        rows = [
            {
                "player_id": 10,
                "season": 2023,
                "age": 28,
                "ip_1": 180.0,
                "ip_2": 170.0,
                "g_1": 30,
                "g_2": 28,
                "gs_1": 30,
                "gs_2": 28,
                "so_1": 200.0,
                "so_2": 180.0,
                "so_wavg": 1.07,
                "weighted_pt": 880.0,
                "league_so_rate": 0.9,
            },
        ]
        result = rows_to_marcel_inputs(rows, ["so"], lags=2, pitcher=True)
        inp = result[10]
        assert inp.seasons[0].ip == 180.0
        assert inp.seasons[0].g == 30
        assert inp.seasons[0].gs == 30
        assert inp.seasons[0].stats["so"] == 200.0
        assert inp.seasons[1].ip == 170.0

    def test_uses_most_recent_row_per_player(self) -> None:
        rows = [
            {
                "player_id": 1,
                "season": 2022,
                "age": 28,
                "pa_1": 500,
                "hr_1": 25.0,
                "hr_wavg": 0.04,
                "weighted_pt": 2500.0,
                "league_hr_rate": 0.03,
            },
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "hr_1": 30.0,
                "hr_wavg": 0.05,
                "weighted_pt": 3000.0,
                "league_hr_rate": 0.035,
            },
        ]
        result = rows_to_marcel_inputs(rows, ["hr"], lags=1)
        inp = result[1]
        assert inp.age == 29
        assert inp.weighted_pt == pytest.approx(3000.0)

    def test_missing_derived_columns_default_to_zero(self) -> None:
        rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "pa_1": 600,
                "hr_1": 30.0,
            },
        ]
        result = rows_to_marcel_inputs(rows, ["hr"], lags=1)
        inp = result[1]
        assert inp.weighted_rates == {"hr": 0.0}
        assert inp.weighted_pt == 0.0
        assert inp.league_rates == {"hr": 0.0}


class TestProjectionToDomain:
    def test_batter_projection(self) -> None:
        proj = MarcelProjection(
            player_id=1,
            projected_season=2024,
            age=32,
            stats={"hr": 25.0, "h": 140.0},
            rates={"hr": 0.045, "h": 0.25},
            pa=550,
        )
        domain = projection_to_domain(proj, version="v1", player_type="batter")
        assert isinstance(domain, Projection)
        assert domain.player_id == 1
        assert domain.season == 2024
        assert domain.system == "marcel"
        assert domain.version == "v1"
        assert domain.player_type == "batter"
        assert domain.stat_json["hr"] == 25.0
        assert domain.stat_json["pa"] == 550
        assert "rates" not in domain.stat_json

    def test_pitcher_projection(self) -> None:
        proj = MarcelProjection(
            player_id=2,
            projected_season=2024,
            age=28,
            stats={"so": 180.0},
            rates={"so": 1.0},
            ip=180.0,
        )
        domain = projection_to_domain(proj, version="v2", player_type="pitcher")
        assert domain.player_type == "pitcher"
        assert domain.stat_json["ip"] == 180.0
        assert domain.stat_json["so"] == 180.0

    def test_batter_rate_stats(self) -> None:
        proj = MarcelProjection(
            player_id=1,
            projected_season=2024,
            age=29,
            stats={
                "h": 150.0,
                "doubles": 30.0,
                "triples": 3.0,
                "hr": 25.0,
                "bb": 60.0,
                "hbp": 5.0,
                "sf": 5.0,
                "r": 80.0,
                "rbi": 85.0,
                "so": 120.0,
                "sb": 10.0,
                "cs": 3.0,
            },
            rates={},
            pa=600,
        )
        domain = projection_to_domain(proj, version="v1", player_type="batter")
        # ab = pa - bb - hbp - sf = 600 - 60 - 5 - 5 = 530
        ab = 530
        assert domain.stat_json["ab"] == pytest.approx(ab)
        assert domain.stat_json["avg"] == pytest.approx(150.0 / ab)
        # obp = (h + bb + hbp) / (ab + bb + hbp + sf)
        assert domain.stat_json["obp"] == pytest.approx((150 + 60 + 5) / (ab + 60 + 5 + 5))
        # singles = 150 - 30 - 3 - 25 = 92
        # slg = (92 + 2*30 + 3*3 + 4*25) / 530
        assert domain.stat_json["slg"] == pytest.approx((92 + 60 + 9 + 100) / ab)
        assert domain.stat_json["ops"] == pytest.approx(domain.stat_json["obp"] + domain.stat_json["slg"])

    def test_pitcher_rate_stats(self) -> None:
        proj = MarcelProjection(
            player_id=2,
            projected_season=2024,
            age=28,
            stats={
                "w": 12.0,
                "l": 6.0,
                "sv": 0.0,
                "h": 150.0,
                "er": 60.0,
                "hr": 18.0,
                "bb": 45.0,
                "so": 200.0,
            },
            rates={},
            ip=180.0,
        )
        domain = projection_to_domain(proj, version="v1", player_type="pitcher")
        assert domain.stat_json["era"] == pytest.approx(60.0 * 9 / 180.0)
        assert domain.stat_json["whip"] == pytest.approx((150.0 + 45.0) / 180.0)
        assert domain.stat_json["k_per_9"] == pytest.approx(200.0 * 9 / 180.0)
        assert domain.stat_json["bb_per_9"] == pytest.approx(45.0 * 9 / 180.0)

    def test_batter_zero_pa_no_rate_stats(self) -> None:
        proj = MarcelProjection(
            player_id=1,
            projected_season=2024,
            age=29,
            stats={"h": 0.0, "hr": 0.0, "bb": 0.0, "hbp": 0.0, "sf": 0.0},
            rates={},
            pa=0,
        )
        domain = projection_to_domain(proj, version="v1", player_type="batter")
        assert "avg" not in domain.stat_json
        assert "obp" not in domain.stat_json
        assert "slg" not in domain.stat_json
        assert "ops" not in domain.stat_json

    def test_pitcher_zero_ip_no_rate_stats(self) -> None:
        proj = MarcelProjection(
            player_id=2,
            projected_season=2024,
            age=28,
            stats={"er": 0.0, "h": 0.0, "bb": 0.0, "so": 0.0},
            rates={},
            ip=0.0,
        )
        domain = projection_to_domain(proj, version="v1", player_type="pitcher")
        assert "era" not in domain.stat_json
        assert "whip" not in domain.stat_json
        assert "k_per_9" not in domain.stat_json
        assert "bb_per_9" not in domain.stat_json


class TestExtractPtFromRows:
    def test_extracts_column_value(self) -> None:
        rows = [{"player_id": 1, "season": 2023, "proj_pa": 500}]
        result = extract_pt_from_rows(rows, "proj_pa")
        assert result == {1: 500.0}

    def test_uses_most_recent_row_per_player(self) -> None:
        rows = [
            {"player_id": 1, "season": 2022, "proj_pa": 400},
            {"player_id": 1, "season": 2023, "proj_pa": 550},
        ]
        result = extract_pt_from_rows(rows, "proj_pa")
        assert result == {1: 550.0}

    def test_multiple_players(self) -> None:
        rows = [
            {"player_id": 1, "season": 2023, "consensus_pa": 500},
            {"player_id": 2, "season": 2023, "consensus_pa": 600},
        ]
        result = extract_pt_from_rows(rows, "consensus_pa")
        assert result == {1: 500.0, 2: 600.0}

    def test_missing_column_excluded(self) -> None:
        rows = [{"player_id": 1, "season": 2023}]
        result = extract_pt_from_rows(rows, "proj_pa")
        assert result == {}

    def test_zero_value_excluded(self) -> None:
        rows = [{"player_id": 1, "season": 2023, "proj_pa": 0}]
        result = extract_pt_from_rows(rows, "proj_pa")
        assert result == {}

    def test_nan_value_excluded(self) -> None:
        rows = [{"player_id": 1, "season": 2023, "consensus_pa": float("nan")}]
        result = extract_pt_from_rows(rows, "consensus_pa")
        assert result == {}

    def test_none_value_excluded(self) -> None:
        rows = [{"player_id": 1, "season": 2023, "proj_pa": None}]
        result = extract_pt_from_rows(rows, "proj_pa")
        assert result == {}

    def test_pitcher_ip_column(self) -> None:
        rows = [{"player_id": 10, "season": 2023, "proj_ip": 180.0}]
        result = extract_pt_from_rows(rows, "proj_ip")
        assert result == {10: 180.0}
