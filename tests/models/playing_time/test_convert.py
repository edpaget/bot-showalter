from fantasy_baseball_manager.models.playing_time.convert import pt_projection_to_domain


class TestPtProjectionToDomain:
    def test_batter_returns_only_pa(self) -> None:
        proj = pt_projection_to_domain(
            player_id=1,
            projected_season=2025,
            pt=580.0,
            pitcher=False,
            version="v1",
        )
        assert proj.stat_json == {"pa": 580}
        assert "ip" not in proj.stat_json

    def test_batter_pa_is_int(self) -> None:
        proj = pt_projection_to_domain(
            player_id=1,
            projected_season=2025,
            pt=580.7,
            pitcher=False,
            version="v1",
        )
        assert isinstance(proj.stat_json["pa"], int)
        assert proj.stat_json["pa"] == 581

    def test_batter_pa_rounds_half(self) -> None:
        proj = pt_projection_to_domain(
            player_id=1,
            projected_season=2025,
            pt=580.5,
            pitcher=False,
            version="v1",
        )
        assert proj.stat_json["pa"] == 580

    def test_pitcher_returns_only_ip(self) -> None:
        proj = pt_projection_to_domain(
            player_id=10,
            projected_season=2025,
            pt=185.5,
            pitcher=True,
            version="v1",
        )
        assert proj.stat_json == {"ip": 185.5}
        assert "pa" not in proj.stat_json

    def test_system_is_playing_time(self) -> None:
        proj = pt_projection_to_domain(
            player_id=1,
            projected_season=2025,
            pt=600.0,
            pitcher=False,
            version="v1",
        )
        assert proj.system == "playing_time"

    def test_version_is_passed_through(self) -> None:
        proj = pt_projection_to_domain(
            player_id=1,
            projected_season=2025,
            pt=600.0,
            pitcher=False,
            version="2025.1",
        )
        assert proj.version == "2025.1"

    def test_player_type_batter(self) -> None:
        proj = pt_projection_to_domain(
            player_id=1,
            projected_season=2025,
            pt=600.0,
            pitcher=False,
            version="v1",
        )
        assert proj.player_type == "batter"

    def test_player_type_pitcher(self) -> None:
        proj = pt_projection_to_domain(
            player_id=10,
            projected_season=2025,
            pt=185.5,
            pitcher=True,
            version="v1",
        )
        assert proj.player_type == "pitcher"
