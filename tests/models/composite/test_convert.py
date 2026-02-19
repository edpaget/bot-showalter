import pytest

from fantasy_baseball_manager.models.composite.convert import (
    best_rows_per_player,
    composite_projection_to_domain,
    batter_rates_to_counting,
    extract_projected_pt,
    pitcher_rates_to_counting,
)


class TestBestRowsPerPlayer:
    def test_single_row(self) -> None:
        rows = [{"player_id": 1, "season": 2023, "proj_pa": 600}]
        result = best_rows_per_player(rows)
        assert result == {1: {"player_id": 1, "season": 2023, "proj_pa": 600}}

    def test_multiple_seasons_takes_latest(self) -> None:
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

    def test_empty(self) -> None:
        result = best_rows_per_player([])
        assert result == {}


class TestBatterRatesToCounting:
    def test_basic_conversion_at_600_pa(self) -> None:
        rates = {"avg": 0.280, "obp": 0.350, "slg": 0.450, "woba": 0.340}
        result = batter_rates_to_counting(rates, 600)
        # ab = pa * (1 - bb_frac - hbp_frac - sf_frac)
        # default bb_frac=0.085, hbp_frac=0.012, sf_frac=0.01
        ab = result["ab"]
        assert result["h"] == pytest.approx(0.280 * ab)
        assert result["pa"] == 600

    def test_zero_pa_returns_zeros(self) -> None:
        rates = {"avg": 0.280, "obp": 0.350, "slg": 0.450}
        result = batter_rates_to_counting(rates, 0)
        assert result["h"] == 0.0
        assert result["hr"] == 0.0
        assert result["ab"] == 0.0

    def test_h_equals_avg_times_ab(self) -> None:
        rates = {"avg": 0.300, "obp": 0.400, "slg": 0.500}
        result = batter_rates_to_counting(rates, 500)
        assert result["h"] == pytest.approx(0.300 * result["ab"])


class TestPitcherRatesToCounting:
    def test_basic_conversion_at_180_ip(self) -> None:
        rates = {"era": 3.00, "k_per_9": 9.0, "bb_per_9": 3.0, "hr_per_9": 1.0, "whip": 1.20}
        result = pitcher_rates_to_counting(rates, 180.0)
        assert result["er"] == pytest.approx(3.00 * 180 / 9)  # 60
        assert result["so"] == pytest.approx(9.0 * 180 / 9)  # 180
        assert result["bb"] == pytest.approx(3.0 * 180 / 9)  # 60
        assert result["hr"] == pytest.approx(1.0 * 180 / 9)  # 20
        # h = whip * ip - bb = 1.20 * 180 - 60 = 156
        assert result["h"] == pytest.approx(1.20 * 180 - 60)

    def test_zero_ip_returns_zeros(self) -> None:
        rates = {"era": 3.00, "k_per_9": 9.0, "bb_per_9": 3.0, "hr_per_9": 1.0, "whip": 1.20}
        result = pitcher_rates_to_counting(rates, 0.0)
        assert result == {"er": 0.0, "so": 0.0, "bb": 0.0, "hr": 0.0, "h": 0.0}

    def test_missing_rate_keys_default_to_zero(self) -> None:
        result = pitcher_rates_to_counting({}, 180.0)
        assert result["er"] == 0.0
        assert result["so"] == 0.0
        assert result["bb"] == 0.0
        assert result["hr"] == 0.0
        assert result["h"] == 0.0


class TestExtractProjectedPt:
    def test_extracts_pa_for_batters(self) -> None:
        rows = [
            {"player_id": 1, "season": 2023, "proj_pa": 600},
            {"player_id": 2, "season": 2023, "proj_pa": 500},
        ]
        result = extract_projected_pt(rows, pitcher=False)
        assert result == {1: 600.0, 2: 500.0}

    def test_extracts_ip_for_pitchers(self) -> None:
        rows = [
            {"player_id": 10, "season": 2023, "proj_ip": 180.0},
        ]
        result = extract_projected_pt(rows, pitcher=True)
        assert result == {10: 180.0}

    def test_uses_best_row_per_player(self) -> None:
        rows = [
            {"player_id": 1, "season": 2022, "proj_pa": 550},
            {"player_id": 1, "season": 2023, "proj_pa": 600},
        ]
        result = extract_projected_pt(rows, pitcher=False)
        assert result == {1: 600.0}

    def test_missing_pt_defaults_to_zero(self) -> None:
        rows = [
            {"player_id": 1, "season": 2023},
        ]
        result = extract_projected_pt(rows, pitcher=False)
        assert result == {1: 0.0}

    def test_empty_rows(self) -> None:
        result = extract_projected_pt([], pitcher=False)
        assert result == {}


class TestCompositeProjectionToDomain:
    def test_batter_projection(self) -> None:
        proj = composite_projection_to_domain(
            player_id=1,
            projected_season=2025,
            stats={"hr": 35.5, "bb": 60.2},
            rates={"hr": 0.06, "bb": 0.10},
            pt=600,
            pitcher=False,
            version="v1",
        )
        assert proj.system == "composite"
        assert proj.player_type == "batter"
        assert proj.stat_json["hr"] == 35.5
        assert proj.stat_json["bb"] == 60.2
        assert proj.stat_json["rates"] == {"hr": 0.06, "bb": 0.10}
        assert proj.stat_json["pa"] == 600

    def test_pitcher_projection(self) -> None:
        proj = composite_projection_to_domain(
            player_id=10,
            projected_season=2025,
            stats={"so": 200.0},
            rates={"so": 1.1},
            pt=180.0,
            pitcher=True,
            version="v1",
        )
        assert proj.system == "composite"
        assert proj.player_type == "pitcher"
        assert proj.stat_json["so"] == 200.0
        assert proj.stat_json["ip"] == 180.0

    def test_batter_projection_custom_system(self) -> None:
        proj = composite_projection_to_domain(
            player_id=1,
            projected_season=2025,
            stats={"hr": 35.5},
            rates={"hr": 0.06},
            pt=600,
            pitcher=False,
            version="v1",
            system="composite-mle",
        )
        assert proj.system == "composite-mle"

    def test_system_defaults_to_composite(self) -> None:
        proj = composite_projection_to_domain(
            player_id=1,
            projected_season=2025,
            stats={"hr": 35.5},
            rates={"hr": 0.06},
            pt=600,
            pitcher=False,
            version="v1",
        )
        assert proj.system == "composite"

    def test_includes_pt_system_metadata(self) -> None:
        proj = composite_projection_to_domain(
            player_id=1,
            projected_season=2025,
            stats={"hr": 35.5},
            rates={"hr": 0.06},
            pt=600,
            pitcher=False,
            version="v1",
        )
        assert proj.stat_json["_pt_system"] == "playing_time"

    def test_batter_rate_stats(self) -> None:
        stats = {
            "h": 150.0,
            "doubles": 30.0,
            "triples": 3.0,
            "hr": 25.0,
            "bb": 60.0,
            "hbp": 5.0,
            "sf": 5.0,
        }
        pa = 600
        proj = composite_projection_to_domain(
            player_id=1,
            projected_season=2025,
            stats=stats,
            rates={},
            pt=pa,
            pitcher=False,
            version="v1",
        )
        sj = proj.stat_json
        ab = pa - 60 - 5 - 5  # 530
        assert sj["ab"] == ab
        assert sj["avg"] == 150 / ab
        assert sj["obp"] == (150 + 60 + 5) / (ab + 60 + 5 + 5)
        singles = 150 - 30 - 3 - 25  # 92
        slg = (singles + 2 * 30 + 3 * 3 + 4 * 25) / ab
        assert sj["slg"] == slg
        assert sj["ops"] == sj["obp"] + sj["slg"]

    def test_pitcher_rate_stats(self) -> None:
        stats = {"er": 60.0, "h": 150.0, "bb": 50.0, "so": 180.0}
        ip = 180.0
        proj = composite_projection_to_domain(
            player_id=10,
            projected_season=2025,
            stats=stats,
            rates={},
            pt=ip,
            pitcher=True,
            version="v1",
        )
        sj = proj.stat_json
        assert sj["era"] == 60 * 9 / 180
        assert sj["whip"] == (150 + 50) / 180
        assert sj["k_per_9"] == 180 * 9 / 180
        assert sj["bb_per_9"] == 50 * 9 / 180

    def test_batter_zero_pa_no_rate_stats(self) -> None:
        proj = composite_projection_to_domain(
            player_id=1,
            projected_season=2025,
            stats={"h": 0.0, "hr": 0.0},
            rates={},
            pt=0,
            pitcher=False,
            version="v1",
        )
        for key in ("avg", "obp", "slg", "ops", "ab"):
            assert key not in proj.stat_json

    def test_pitcher_zero_ip_no_rate_stats(self) -> None:
        proj = composite_projection_to_domain(
            player_id=10,
            projected_season=2025,
            stats={"er": 0.0, "so": 0.0},
            rates={},
            pt=0.0,
            pitcher=True,
            version="v1",
        )
        for key in ("era", "whip", "k_per_9", "bb_per_9"):
            assert key not in proj.stat_json
