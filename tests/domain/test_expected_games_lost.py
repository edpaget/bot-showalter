import pytest

from fantasy_baseball_manager.domain import ExpectedGamesLost


class TestExpectedGamesLost:
    def test_construction(self) -> None:
        egl = ExpectedGamesLost(
            player_id=1,
            expected_days_lost=25.0,
            p_full_season=0.75,
            confidence="medium",
        )
        assert egl.player_id == 1
        assert egl.expected_days_lost == 25.0
        assert egl.p_full_season == 0.75
        assert egl.confidence == "medium"

    def test_frozen(self) -> None:
        egl = ExpectedGamesLost(
            player_id=1,
            expected_days_lost=25.0,
            p_full_season=0.75,
            confidence="medium",
        )
        with pytest.raises(AttributeError):
            egl.expected_days_lost = 30.0  # type: ignore[misc]
