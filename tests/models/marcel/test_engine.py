import pytest

from fantasy_baseball_manager.models.marcel.types import (
    LeagueAverages,
    MarcelConfig,
    MarcelInput,
    SeasonLine,
)
from fantasy_baseball_manager.models.marcel.engine import (
    age_adjust,
    project_all,
    project_player,
    project_playing_time,
    regress_to_mean,
)


class TestRegressToMean:
    def test_known_inputs(self) -> None:
        rates = {"hr": 0.05, "h": 0.27}
        league = LeagueAverages(rates={"hr": 0.03, "h": 0.25})
        # (rate * pt + league * n) / (pt + n)
        # hr: (0.05 * 600 + 0.03 * 1200) / (600 + 1200) = (30+36)/1800 = 66/1800
        result = regress_to_mean(rates, league, playing_time=600.0, regression_n=1200.0)
        assert result["hr"] == pytest.approx(66.0 / 1800.0)
        assert result["h"] == pytest.approx((0.27 * 600 + 0.25 * 1200) / 1800.0)

    def test_few_pa_regresses_toward_league(self) -> None:
        rates = {"hr": 0.10}
        league = LeagueAverages(rates={"hr": 0.03})
        result = regress_to_mean(rates, league, playing_time=50.0, regression_n=1200.0)
        # 50 PA vs 1200 regression — heavily regressed
        # (0.10 * 50 + 0.03 * 1200) / (50 + 1200) = (5+36)/1250
        expected = (5.0 + 36.0) / 1250.0
        assert result["hr"] == pytest.approx(expected)
        # Should be much closer to league average
        assert abs(result["hr"] - 0.03) < abs(result["hr"] - 0.10)

    def test_many_pa_stays_near_own_rate(self) -> None:
        rates = {"hr": 0.05}
        league = LeagueAverages(rates={"hr": 0.03})
        result = regress_to_mean(rates, league, playing_time=5000.0, regression_n=1200.0)
        # With 5000 PA, own rate dominates
        assert abs(result["hr"] - 0.05) < abs(result["hr"] - 0.03)


class TestProjectPlayingTime:
    def test_batter_three_years(self) -> None:
        seasons = [
            SeasonLine(stats={}, pa=600),
            SeasonLine(stats={}, pa=500),
            SeasonLine(stats={}, pa=400),
        ]
        config = MarcelConfig()
        # w1 * yr1 + w2 * yr2 + baseline = 0.5*600 + 0.1*500 + 200
        result = project_playing_time(seasons, config)
        assert result == pytest.approx(0.5 * 600 + 0.1 * 500 + 200)

    def test_batter_two_years(self) -> None:
        seasons = [
            SeasonLine(stats={}, pa=600),
            SeasonLine(stats={}, pa=500),
        ]
        config = MarcelConfig()
        # w1 * yr1 + w2 * yr2 + baseline
        result = project_playing_time(seasons, config)
        assert result == pytest.approx(0.5 * 600 + 0.1 * 500 + 200)

    def test_batter_one_year(self) -> None:
        seasons = [
            SeasonLine(stats={}, pa=600),
        ]
        config = MarcelConfig()
        # w1 * yr1 + 0 + baseline
        result = project_playing_time(seasons, config)
        assert result == pytest.approx(0.5 * 600 + 200)

    def test_pitcher_starter(self) -> None:
        seasons = [
            SeasonLine(stats={}, ip=180.0, g=30, gs=30),
            SeasonLine(stats={}, ip=170.0, g=28, gs=28),
            SeasonLine(stats={}, ip=160.0, g=26, gs=26),
        ]
        config = MarcelConfig()
        # gs/g = 1.0 >= 0.5 → starter baseline (60)
        # w1 * yr1 + w2 * yr2 + baseline = 0.5*180 + 0.1*170 + 60
        result = project_playing_time(seasons, config)
        assert result == pytest.approx(0.5 * 180 + 0.1 * 170 + 60)

    def test_pitcher_reliever(self) -> None:
        seasons = [
            SeasonLine(stats={}, ip=70.0, g=60, gs=0),
            SeasonLine(stats={}, ip=65.0, g=55, gs=0),
            SeasonLine(stats={}, ip=60.0, g=50, gs=0),
        ]
        config = MarcelConfig()
        # gs/g = 0/60 = 0 < 0.5 → reliever baseline (25)
        result = project_playing_time(seasons, config)
        assert result == pytest.approx(0.5 * 70 + 0.1 * 65 + 25)

    def test_empty_seasons_returns_baseline(self) -> None:
        config = MarcelConfig()
        result = project_playing_time([], config)
        assert result == pytest.approx(200.0)  # batter baseline default


class TestAgeAdjust:
    def test_young_player_improves(self) -> None:
        rates = {"hr": 0.05, "h": 0.25}
        config = MarcelConfig()  # peak=29, improvement=0.006
        # age 25 → 4 years below peak → factor = 1 + 4 * 0.006 = 1.024
        result = age_adjust(rates, age=25, config=config)
        assert result["hr"] == pytest.approx(0.05 * 1.024)
        assert result["h"] == pytest.approx(0.25 * 1.024)

    def test_old_player_declines(self) -> None:
        rates = {"hr": 0.05}
        config = MarcelConfig()  # peak=29, decline=0.003
        # age 35 → 6 years above peak → factor = 1 - 6 * 0.003 = 0.982
        result = age_adjust(rates, age=35, config=config)
        assert result["hr"] == pytest.approx(0.05 * 0.982)

    def test_peak_age_unchanged(self) -> None:
        rates = {"hr": 0.05}
        config = MarcelConfig()  # peak=29
        result = age_adjust(rates, age=29, config=config)
        assert result["hr"] == pytest.approx(0.05)


class TestProjectPlayer:
    def test_end_to_end_batter(self) -> None:
        seasons = (
            SeasonLine(stats={"hr": 30.0}, pa=600),
            SeasonLine(stats={"hr": 25.0}, pa=500),
            SeasonLine(stats={"hr": 20.0}, pa=400),
        )
        marcel_input = MarcelInput(
            weighted_rates={"hr": 310.0 / 6200.0},
            weighted_pt=6200.0,
            league_rates={"hr": 0.03},
            age=29,
            seasons=seasons,
        )
        config = MarcelConfig(batting_categories=("hr",))
        proj = project_player(
            player_id=1,
            marcel_input=marcel_input,
            projected_season=2024,
            config=config,
        )
        assert proj.player_id == 1
        assert proj.projected_season == 2024
        assert proj.age == 29
        assert proj.pa > 0
        assert "hr" in proj.stats
        assert "hr" in proj.rates
        assert proj.stats["hr"] > 0

    def test_end_to_end_pitcher(self) -> None:
        seasons = (
            SeasonLine(stats={"so": 200.0}, ip=180.0, g=30, gs=30),
            SeasonLine(stats={"so": 180.0}, ip=170.0, g=28, gs=28),
            SeasonLine(stats={"so": 150.0}, ip=160.0, g=26, gs=26),
        )
        marcel_input = MarcelInput(
            weighted_rates={"so": 1110.0 / 1040.0},
            weighted_pt=1040.0,
            league_rates={"so": 0.9},
            age=28,
            seasons=seasons,
        )
        config = MarcelConfig(pitching_categories=("so",))
        proj = project_player(
            player_id=2,
            marcel_input=marcel_input,
            projected_season=2024,
            config=config,
        )
        assert proj.player_id == 2
        assert proj.ip > 0
        assert proj.pa == 0
        assert "so" in proj.stats

    def test_one_year_player(self) -> None:
        seasons = (SeasonLine(stats={"hr": 10.0}, pa=200),)
        marcel_input = MarcelInput(
            weighted_rates={"hr": 10.0 / 200.0},
            weighted_pt=200.0 * 5.0,
            league_rates={"hr": 0.03},
            age=25,
            seasons=seasons,
        )
        config = MarcelConfig(batting_categories=("hr",))
        proj = project_player(
            player_id=3,
            marcel_input=marcel_input,
            projected_season=2024,
            config=config,
        )
        assert proj.player_id == 3
        # With only 1000 weighted PT vs 1200 regression, should be regressed
        assert proj.rates["hr"] < 10.0 / 200.0


class TestProjectPlayerWithProjectedPt:
    def test_uses_supplied_value_for_batter(self) -> None:
        seasons = (SeasonLine(stats={"hr": 30.0}, pa=600),)
        marcel_input = MarcelInput(
            weighted_rates={"hr": 0.05},
            weighted_pt=3000.0,
            league_rates={"hr": 0.03},
            age=29,
            seasons=seasons,
        )
        config = MarcelConfig(batting_categories=("hr",))
        proj = project_player(
            player_id=1,
            marcel_input=marcel_input,
            projected_season=2024,
            config=config,
            projected_pt=450.0,
        )
        assert proj.pa == 450

    def test_without_projected_pt_uses_internal_formula(self) -> None:
        seasons = (SeasonLine(stats={"hr": 30.0}, pa=600),)
        marcel_input = MarcelInput(
            weighted_rates={"hr": 0.05},
            weighted_pt=3000.0,
            league_rates={"hr": 0.03},
            age=29,
            seasons=seasons,
        )
        config = MarcelConfig(batting_categories=("hr",))
        proj = project_player(
            player_id=1,
            marcel_input=marcel_input,
            projected_season=2024,
            config=config,
        )
        expected_pt = project_playing_time(seasons, config)
        assert proj.pa == int(expected_pt)

    def test_uses_supplied_value_for_pitcher(self) -> None:
        seasons = (SeasonLine(stats={"so": 200.0}, ip=180.0, g=30, gs=30),)
        marcel_input = MarcelInput(
            weighted_rates={"so": 1.1},
            weighted_pt=900.0,
            league_rates={"so": 0.9},
            age=28,
            seasons=seasons,
        )
        config = MarcelConfig(pitching_categories=("so",))
        proj = project_player(
            player_id=2,
            marcel_input=marcel_input,
            projected_season=2024,
            config=config,
            projected_pt=180.0,
        )
        assert proj.ip == 180.0


class TestProjectAll:
    def test_with_projected_pts_applies_to_matching_player(self) -> None:
        players: dict[int, MarcelInput] = {
            1: MarcelInput(
                weighted_rates={"hr": 0.05},
                weighted_pt=3000.0,
                league_rates={"hr": 0.03},
                age=29,
                seasons=(SeasonLine(stats={"hr": 30.0}, pa=600),),
            ),
            2: MarcelInput(
                weighted_rates={"hr": 0.04},
                weighted_pt=2500.0,
                league_rates={"hr": 0.03},
                age=25,
                seasons=(SeasonLine(stats={"hr": 20.0}, pa=500),),
            ),
        }
        config = MarcelConfig(batting_categories=("hr",))
        results = project_all(players, 2024, config, projected_pts={1: 450.0})
        by_id = {p.player_id: p for p in results}
        # Player 1 uses supplied PT
        assert by_id[1].pa == 450
        # Player 2 falls back to internal formula
        expected_pt = project_playing_time(players[2].seasons, config)
        assert by_id[2].pa == int(expected_pt)

    def test_without_projected_pts_uses_internal_formula(self) -> None:
        players: dict[int, MarcelInput] = {
            1: MarcelInput(
                weighted_rates={"hr": 0.05},
                weighted_pt=3000.0,
                league_rates={"hr": 0.03},
                age=29,
                seasons=(SeasonLine(stats={"hr": 30.0}, pa=600),),
            ),
        }
        config = MarcelConfig(batting_categories=("hr",))
        results = project_all(players, 2024, config)
        expected_pt = project_playing_time(players[1].seasons, config)
        assert results[0].pa == int(expected_pt)

    def test_batch_multiple_players(self) -> None:
        players: dict[int, MarcelInput] = {
            1: MarcelInput(
                weighted_rates={"hr": 0.05},
                weighted_pt=3000.0,
                league_rates={"hr": 0.03},
                age=29,
                seasons=(SeasonLine(stats={"hr": 30.0}, pa=600),),
            ),
            2: MarcelInput(
                weighted_rates={"hr": 0.04},
                weighted_pt=2500.0,
                league_rates={"hr": 0.03},
                age=25,
                seasons=(SeasonLine(stats={"hr": 20.0}, pa=500),),
            ),
        }
        config = MarcelConfig(batting_categories=("hr",))
        results = project_all(players, 2024, config)
        assert len(results) == 2
        ids = {p.player_id for p in results}
        assert ids == {1, 2}
