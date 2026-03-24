import pytest

from fantasy_baseball_manager.domain.feature_candidate import (
    CandidateValue,
    FeatureCandidate,
)
from fantasy_baseball_manager.domain.identity import PlayerType


class TestFeatureCandidate:
    def test_construction(self) -> None:
        fc = FeatureCandidate(
            name="barrel_ev",
            expression="AVG(launch_speed) FILTER (WHERE barrel = 1)",
            player_type=PlayerType.BATTER,
            min_pa=100,
            min_ip=None,
            created_at="2026-03-02",
        )
        assert fc.name == "barrel_ev"
        assert fc.expression == "AVG(launch_speed) FILTER (WHERE barrel = 1)"
        assert fc.player_type == "batter"
        assert fc.min_pa == 100
        assert fc.min_ip is None
        assert fc.created_at == "2026-03-02"

    def test_frozen_immutability(self) -> None:
        fc = FeatureCandidate(
            name="barrel_ev",
            expression="AVG(launch_speed)",
            player_type=PlayerType.BATTER,
            min_pa=None,
            min_ip=None,
            created_at="2026-03-02",
        )
        with pytest.raises(AttributeError):
            fc.name = "other"  # type: ignore[misc]

    def test_none_fields(self) -> None:
        fc = FeatureCandidate(
            name="test",
            expression="COUNT(*)",
            player_type=PlayerType.PITCHER,
            min_pa=None,
            min_ip=None,
            created_at="2026-03-02",
        )
        assert fc.min_pa is None
        assert fc.min_ip is None

    def test_pitcher_with_min_ip(self) -> None:
        fc = FeatureCandidate(
            name="k_rate",
            expression="COUNT(*) FILTER (WHERE description = 'swinging_strike') * 1.0 / COUNT(*)",
            player_type=PlayerType.PITCHER,
            min_pa=None,
            min_ip=50.0,
            created_at="2026-03-02",
        )
        assert fc.min_ip == 50.0


class TestCandidateValue:
    def test_construction(self) -> None:
        cv = CandidateValue(player_id=12345, season=2023, value=0.345)
        assert cv.player_id == 12345
        assert cv.season == 2023
        assert cv.value == 0.345

    def test_frozen_immutability(self) -> None:
        cv = CandidateValue(player_id=1, season=2023, value=1.0)
        with pytest.raises(AttributeError):
            cv.value = 2.0  # type: ignore[misc]

    def test_none_value(self) -> None:
        cv = CandidateValue(player_id=1, season=2023, value=None)
        assert cv.value is None
