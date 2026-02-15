from fantasy_baseball_manager.domain.projection import Projection
from fantasy_baseball_manager.models.marcel.convert import (
    projection_to_domain,
    rows_to_player_seasons,
)
from fantasy_baseball_manager.models.marcel.types import MarcelProjection


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
        assert domain.stat_json["rates"]["hr"] == 0.045

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
