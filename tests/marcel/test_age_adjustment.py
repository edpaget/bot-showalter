import pytest

from fantasy_baseball_manager.marcel.age_adjustment import age_multiplier


class TestAgeMultiplier:
    def test_peak_age_29(self) -> None:
        assert age_multiplier(29) == 1.0

    def test_young_player_25(self) -> None:
        # (29 - 25) * 0.006 = 0.024
        assert age_multiplier(25) == pytest.approx(1.024)

    def test_young_player_22(self) -> None:
        # (29 - 22) * 0.006 = 0.042
        assert age_multiplier(22) == pytest.approx(1.042)

    def test_old_player_33(self) -> None:
        # (29 - 33) * 0.003 = -0.012
        assert age_multiplier(33) == pytest.approx(0.988)

    def test_old_player_38(self) -> None:
        # (29 - 38) * 0.003 = -0.027
        assert age_multiplier(38) == pytest.approx(0.973)

    def test_one_year_young(self) -> None:
        assert age_multiplier(28) == pytest.approx(1.006)

    def test_one_year_old(self) -> None:
        assert age_multiplier(30) == pytest.approx(0.997)
