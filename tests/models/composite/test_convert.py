from fantasy_baseball_manager.models.composite.convert import (
    composite_projection_to_domain,
    extract_projected_pt,
)


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
