import pytest

from fantasy_baseball_manager.models.marcel.types import (
    LeagueAverages,
    MarcelConfig,
    SeasonLine,
)
from fantasy_baseball_manager.models.marcel.engine import (
    age_adjust,
    compute_league_averages,
    project_all,
    project_player,
    project_playing_time,
    regress_to_mean,
    weighted_average_rates,
)


class TestWeightedAverageRates:
    def test_three_year_batter(self) -> None:
        seasons = [
            SeasonLine(stats={"hr": 30.0, "h": 150.0}, pa=600),
            SeasonLine(stats={"hr": 25.0, "h": 140.0}, pa=500),
            SeasonLine(stats={"hr": 20.0, "h": 120.0}, pa=400),
        ]
        weights = (5.0, 4.0, 3.0)
        rates = weighted_average_rates(seasons, weights, ["hr", "h"])
        # hr: (30*5 + 25*4 + 20*3) / (600*5 + 500*4 + 400*3)
        # = (150+100+60) / (3000+2000+1200) = 310/6200
        expected_hr = 310.0 / 6200.0
        assert rates["hr"] == pytest.approx(expected_hr)
        # h: (150*5 + 140*4 + 120*3) / 6200
        expected_h = (750.0 + 560.0 + 360.0) / 6200.0
        assert rates["h"] == pytest.approx(expected_h)

    def test_two_year_batter(self) -> None:
        seasons = [
            SeasonLine(stats={"hr": 30.0}, pa=600),
            SeasonLine(stats={"hr": 25.0}, pa=500),
        ]
        weights = (5.0, 4.0, 3.0)
        rates = weighted_average_rates(seasons, weights, ["hr"])
        # Only 2 seasons: weights renormalized to (5, 4) but used as-is for weighting
        # hr: (30*5 + 25*4) / (600*5 + 500*4) = 250/5000
        expected_hr = 250.0 / 5000.0
        assert rates["hr"] == pytest.approx(expected_hr)

    def test_one_year_batter(self) -> None:
        seasons = [
            SeasonLine(stats={"hr": 30.0}, pa=600),
        ]
        weights = (5.0, 4.0, 3.0)
        rates = weighted_average_rates(seasons, weights, ["hr"])
        # Only 1 season: weight is just (5,)
        # hr: 30*5 / (600*5) = 30/600
        expected_hr = 30.0 / 600.0
        assert rates["hr"] == pytest.approx(expected_hr)

    def test_zero_pa_season_excluded(self) -> None:
        seasons = [
            SeasonLine(stats={"hr": 30.0}, pa=600),
            SeasonLine(stats={"hr": 0.0}, pa=0),  # missed season
            SeasonLine(stats={"hr": 20.0}, pa=400),
        ]
        weights = (5.0, 4.0, 3.0)
        rates = weighted_average_rates(seasons, weights, ["hr"])
        # 0-PA season contributes 0 weighted stats and 0 weighted PA
        # hr: (30*5 + 0*4 + 20*3) / (600*5 + 0*4 + 400*3) = 210/4200
        expected_hr = 210.0 / 4200.0
        assert rates["hr"] == pytest.approx(expected_hr)

    def test_pitcher_with_ip(self) -> None:
        seasons = [
            SeasonLine(stats={"so": 200.0, "er": 60.0}, ip=180.0, g=30, gs=30),
            SeasonLine(stats={"so": 180.0, "er": 55.0}, ip=170.0, g=28, gs=28),
            SeasonLine(stats={"so": 150.0, "er": 50.0}, ip=160.0, g=26, gs=26),
        ]
        weights = (3.0, 2.0, 1.0)
        rates = weighted_average_rates(seasons, weights, ["so", "er"])
        # so: (200*3 + 180*2 + 150*1) / (180*3 + 170*2 + 160*1)
        # = (600+360+150) / (540+340+160) = 1110/1040
        expected_so = 1110.0 / 1040.0
        assert rates["so"] == pytest.approx(expected_so)

    def test_empty_seasons_returns_zero_rates(self) -> None:
        rates = weighted_average_rates([], (5.0, 4.0, 3.0), ["hr", "h"])
        assert rates["hr"] == 0.0
        assert rates["h"] == 0.0


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


class TestComputeLeagueAverages:
    def test_batting_averages(self) -> None:
        all_seasons: dict[int, list[SeasonLine]] = {
            1: [SeasonLine(stats={"hr": 30.0, "h": 150.0}, pa=600)],
            2: [SeasonLine(stats={"hr": 20.0, "h": 130.0}, pa=500)],
        }
        result = compute_league_averages(all_seasons, ["hr", "h"])
        # total hr = 30 + 20 = 50, total PA = 600 + 500 = 1100
        assert result.rates["hr"] == pytest.approx(50.0 / 1100.0)
        assert result.rates["h"] == pytest.approx(280.0 / 1100.0)

    def test_pitcher_averages(self) -> None:
        all_seasons: dict[int, list[SeasonLine]] = {
            1: [SeasonLine(stats={"so": 200.0}, ip=180.0, g=30, gs=30)],
            2: [SeasonLine(stats={"so": 150.0}, ip=170.0, g=28, gs=28)],
        }
        result = compute_league_averages(all_seasons, ["so"])
        assert result.rates["so"] == pytest.approx(350.0 / 350.0)

    def test_multi_season_players(self) -> None:
        all_seasons: dict[int, list[SeasonLine]] = {
            1: [
                SeasonLine(stats={"hr": 30.0}, pa=600),
                SeasonLine(stats={"hr": 25.0}, pa=500),
            ],
        }
        # Uses most recent season (index 0) for league averages
        result = compute_league_averages(all_seasons, ["hr"])
        assert result.rates["hr"] == pytest.approx(30.0 / 600.0)


class TestProjectPlayer:
    def test_end_to_end_batter(self) -> None:
        seasons = [
            SeasonLine(stats={"hr": 30.0}, pa=600),
            SeasonLine(stats={"hr": 25.0}, pa=500),
            SeasonLine(stats={"hr": 20.0}, pa=400),
        ]
        league = LeagueAverages(rates={"hr": 0.03})
        config = MarcelConfig(batting_categories=("hr",))
        proj = project_player(
            player_id=1,
            seasons=seasons,
            age=29,
            projected_season=2024,
            league_avg=league,
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
        seasons = [
            SeasonLine(stats={"so": 200.0}, ip=180.0, g=30, gs=30),
            SeasonLine(stats={"so": 180.0}, ip=170.0, g=28, gs=28),
            SeasonLine(stats={"so": 150.0}, ip=160.0, g=26, gs=26),
        ]
        league = LeagueAverages(rates={"so": 0.9})
        config = MarcelConfig(pitching_categories=("so",))
        proj = project_player(
            player_id=2,
            seasons=seasons,
            age=28,
            projected_season=2024,
            league_avg=league,
            config=config,
        )
        assert proj.player_id == 2
        assert proj.ip > 0
        assert proj.pa == 0
        assert "so" in proj.stats

    def test_one_year_player(self) -> None:
        seasons = [
            SeasonLine(stats={"hr": 10.0}, pa=200),
        ]
        league = LeagueAverages(rates={"hr": 0.03})
        config = MarcelConfig(batting_categories=("hr",))
        proj = project_player(
            player_id=3,
            seasons=seasons,
            age=25,
            projected_season=2024,
            league_avg=league,
            config=config,
        )
        assert proj.player_id == 3
        # With only 200 PA, should be heavily regressed
        assert proj.rates["hr"] < 10.0 / 200.0


class TestProjectAll:
    def test_batch_multiple_players(self) -> None:
        players: dict[int, tuple[list[SeasonLine], int]] = {
            1: ([SeasonLine(stats={"hr": 30.0}, pa=600)], 29),
            2: ([SeasonLine(stats={"hr": 20.0}, pa=500)], 25),
        }
        league = LeagueAverages(rates={"hr": 0.03})
        config = MarcelConfig(batting_categories=("hr",))
        results = project_all(players, 2024, league, config)
        assert len(results) == 2
        ids = {p.player_id for p in results}
        assert ids == {1, 2}
