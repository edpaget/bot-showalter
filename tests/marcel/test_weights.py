import pytest

from fantasy_baseball_manager.marcel.weights import (
    projected_ip,
    projected_pa,
    weighted_rate,
)


class TestWeightedRate:
    def test_three_years_batting(self) -> None:
        # 5*25 + 4*20 + 3*15 = 125+80+45 = 250
        # 5*600 + 4*550 + 3*500 = 3000+2200+1500 = 6700
        # league_rate = 0.04, regression = 1200
        # numerator = 250 + 1200*0.04 = 250 + 48 = 298
        # denominator = 6700 + 1200 = 7900
        # rate = 298 / 7900 ≈ 0.037721519
        result = weighted_rate(
            stats=[25, 20, 15],
            opportunities=[600, 550, 500],
            weights=[5, 4, 3],
            league_rate=0.04,
            regression_pa=1200,
        )
        assert result == pytest.approx(298 / 7900)

    def test_missing_year_contributes_zero(self) -> None:
        # Only most recent year available: stats=[30], opps=[500], weights=[5]
        # numerator = 5*30 + 1200*0.05 = 150 + 60 = 210
        # denominator = 5*500 + 1200 = 2500 + 1200 = 3700
        result = weighted_rate(
            stats=[30],
            opportunities=[500],
            weights=[5],
            league_rate=0.05,
            regression_pa=1200,
        )
        assert result == pytest.approx(210 / 3700)

    def test_two_years(self) -> None:
        # 5*10 + 4*8 = 50+32 = 82
        # 5*400 + 4*350 = 2000+1400 = 3400
        # numerator = 82 + 1200*0.03 = 82 + 36 = 118
        # denominator = 3400 + 1200 = 4600
        result = weighted_rate(
            stats=[10, 8],
            opportunities=[400, 350],
            weights=[5, 4],
            league_rate=0.03,
            regression_pa=1200,
        )
        assert result == pytest.approx(118 / 4600)

    def test_zero_stats_regresses_to_league(self) -> None:
        # All zeros => rate should be close to league_rate
        # numerator = 0 + 1200*0.04 = 48
        # denominator = 0 + 1200 = 1200
        result = weighted_rate(
            stats=[0, 0, 0],
            opportunities=[0, 0, 0],
            weights=[5, 4, 3],
            league_rate=0.04,
            regression_pa=1200,
        )
        assert result == pytest.approx(0.04)

    def test_pitcher_weights(self) -> None:
        # Pitchers use weights 3/2/1, regression ~134 outs
        # 3*50 + 2*40 + 1*30 = 150+80+30 = 260
        # 3*540 + 2*480 + 1*420 = 1620+960+420 = 3000
        # numerator = 260 + 134*0.10 = 260 + 13.4 = 273.4
        # denominator = 3000 + 134 = 3134
        result = weighted_rate(
            stats=[50, 40, 30],
            opportunities=[540, 480, 420],
            weights=[3, 2, 1],
            league_rate=0.10,
            regression_pa=134,
        )
        assert result == pytest.approx(273.4 / 3134)


class TestProjectedPA:
    def test_three_years(self) -> None:
        # 0.5*600 + 0.1*550 = 300 + 55 = 355; + 200 = 555
        assert projected_pa(pa_y1=600, pa_y2=550) == pytest.approx(555)

    def test_one_year(self) -> None:
        # 0.5*400 + 0.1*0 = 200; + 200 = 400
        assert projected_pa(pa_y1=400, pa_y2=0) == pytest.approx(400)


class TestProjectedIP:
    def test_starter(self) -> None:
        # 0.5*180 + 0.1*170 = 90 + 17 = 107; + 60 = 167
        assert projected_ip(ip_y1=180.0, ip_y2=170.0, is_starter=True) == pytest.approx(167.0)

    def test_reliever(self) -> None:
        # 0.5*70 + 0.1*65 = 35 + 6.5 = 41.5; + 25 = 66.5
        assert projected_ip(ip_y1=70.0, ip_y2=65.0, is_starter=False) == pytest.approx(66.5)
