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
