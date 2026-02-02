import pytest

from fantasy_baseball_manager.pipeline.stages.adjusters import (
    MarcelAgingAdjuster,
    RebaselineAdjuster,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates


class TestRebaselineAdjuster:
    def test_identity_when_source_equals_target(self) -> None:
        league_rates = {"hr": 0.03, "bb": 0.08}
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={"hr": 0.04, "bb": 0.10},
            metadata={"avg_league_rates": league_rates, "target_rates": league_rates},
        )
        adjuster = RebaselineAdjuster()
        result = adjuster.adjust([p])
        assert result[0].rates["hr"] == pytest.approx(0.04)
        assert result[0].rates["bb"] == pytest.approx(0.10)

    def test_scales_rates_by_target_over_source(self) -> None:
        avg_rates = {"hr": 0.03}
        target_rates = {"hr": 0.036}  # 20% higher target
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={"hr": 0.04},
            metadata={"avg_league_rates": avg_rates, "target_rates": target_rates},
        )
        adjuster = RebaselineAdjuster()
        result = adjuster.adjust([p])
        # 0.04 * (0.036 / 0.03) = 0.04 * 1.2 = 0.048
        assert result[0].rates["hr"] == pytest.approx(0.048)

    def test_preserves_metadata(self) -> None:
        league_rates = {"hr": 0.03}
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={"hr": 0.04},
            metadata={"avg_league_rates": league_rates, "target_rates": league_rates, "extra": 42},
        )
        adjuster = RebaselineAdjuster()
        result = adjuster.adjust([p])
        assert result[0].metadata["extra"] == 42


class TestMarcelAgingAdjuster:
    def test_peak_age_no_adjustment(self) -> None:
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=29,
            rates={"hr": 0.04},
        )
        adjuster = MarcelAgingAdjuster()
        result = adjuster.adjust([p])
        assert result[0].rates["hr"] == pytest.approx(0.04)

    def test_young_player_gets_boost(self) -> None:
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=25,
            rates={"hr": 0.04},
        )
        adjuster = MarcelAgingAdjuster()
        result = adjuster.adjust([p])
        # 1.0 + (29-25)*0.006 = 1.024
        assert result[0].rates["hr"] == pytest.approx(0.04 * 1.024)

    def test_old_player_gets_penalty(self) -> None:
        p = PlayerRates(
            player_id="p1",
            name="Test",
            year=2025,
            age=35,
            rates={"hr": 0.04},
        )
        adjuster = MarcelAgingAdjuster()
        result = adjuster.adjust([p])
        # 1.0 + (29-35)*0.003 = 1.0 - 0.018 = 0.982
        assert result[0].rates["hr"] == pytest.approx(0.04 * 0.982)
