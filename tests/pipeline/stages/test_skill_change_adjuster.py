"""Tests for skill change adjuster."""

import pytest

from fantasy_baseball_manager.pipeline.skill_data import (
    BatterSkillDelta,
    BatterSkillStats,
    PitcherSkillDelta,
    PitcherSkillStats,
)
from fantasy_baseball_manager.pipeline.stages.skill_change_adjuster import (
    SkillChangeAdjuster,
    SkillChangeConfig,
    SkillDeltaComputer,
)
from fantasy_baseball_manager.pipeline.types import PlayerRates
from tests.conftest import make_test_feature_store


class FakeSkillDataSource:
    """Test double for SkillDataSource."""

    def __init__(
        self,
        batter_data: dict[int, list[BatterSkillStats]] | None = None,
        pitcher_data: dict[int, list[PitcherSkillStats]] | None = None,
    ) -> None:
        self._batter_data = batter_data or {}
        self._pitcher_data = pitcher_data or {}

    def batter_skill_stats(self, year: int) -> list[BatterSkillStats]:
        return self._batter_data.get(year, [])

    def pitcher_skill_stats(self, year: int) -> list[PitcherSkillStats]:
        return self._pitcher_data.get(year, [])


class FakeDeltaComputer:
    """Test double for SkillDeltaComputer."""

    def __init__(
        self,
        batter_deltas: dict[str, BatterSkillDelta] | None = None,
        pitcher_deltas: dict[str, PitcherSkillDelta] | None = None,
    ) -> None:
        self._batter_deltas = batter_deltas or {}
        self._pitcher_deltas = pitcher_deltas or {}

    def compute_batter_deltas(self, year: int) -> dict[str, BatterSkillDelta]:
        return self._batter_deltas

    def compute_pitcher_deltas(self, year: int) -> dict[str, PitcherSkillDelta]:
        return self._pitcher_deltas


def make_batter(
    player_id: str = "123",
    name: str = "Test Batter",
    rates: dict[str, float] | None = None,
) -> PlayerRates:
    """Create a test batter PlayerRates."""
    return PlayerRates(
        player_id=player_id,
        name=name,
        year=2025,
        age=28,
        rates=rates or {"hr": 0.04, "doubles": 0.05, "bb": 0.08, "so": 0.20, "sb": 0.02},
        opportunities=600.0,
        metadata={"pa_per_year": [550.0, 580.0, 600.0]},
    )


def make_pitcher(
    player_id: str = "456",
    name: str = "Test Pitcher",
    rates: dict[str, float] | None = None,
) -> PlayerRates:
    """Create a test pitcher PlayerRates."""
    return PlayerRates(
        player_id=player_id,
        name=name,
        year=2025,
        age=30,
        rates=rates or {"hr": 0.03, "so": 0.25, "bb": 0.07, "er": 0.10},
        opportunities=180.0,
        metadata={"ip_per_year": [170.0, 175.0, 180.0]},
    )


def make_batter_delta(
    player_id: str = "123",
    barrel_rate_delta: float | None = 0.0,
    hard_hit_rate_delta: float | None = 0.0,
    exit_velo_max_delta: float | None = 0.0,
    chase_rate_delta: float | None = 0.0,
    whiff_rate_delta: float | None = 0.0,
    sprint_speed_delta: float | None = 0.0,
    pa_current: int = 500,
    pa_prior: int = 450,
) -> BatterSkillDelta:
    """Create a test BatterSkillDelta."""
    return BatterSkillDelta(
        player_id=player_id,
        name="Test Batter",
        year=2025,
        barrel_rate_delta=barrel_rate_delta,
        hard_hit_rate_delta=hard_hit_rate_delta,
        exit_velo_avg_delta=0.0,
        exit_velo_max_delta=exit_velo_max_delta,
        chase_rate_delta=chase_rate_delta,
        whiff_rate_delta=whiff_rate_delta,
        sprint_speed_delta=sprint_speed_delta,
        pa_current=pa_current,
        pa_prior=pa_prior,
    )


def make_pitcher_delta(
    player_id: str = "456",
    fastball_velo_delta: float | None = 0.0,
    whiff_rate_delta: float | None = 0.0,
    gb_rate_delta: float | None = 0.0,
    barrel_rate_against_delta: float | None = 0.0,
    pa_against_current: int = 700,
    pa_against_prior: int = 680,
) -> PitcherSkillDelta:
    """Create a test PitcherSkillDelta."""
    return PitcherSkillDelta(
        player_id=player_id,
        name="Test Pitcher",
        year=2025,
        fastball_velo_delta=fastball_velo_delta,
        whiff_rate_delta=whiff_rate_delta,
        gb_rate_delta=gb_rate_delta,
        barrel_rate_against_delta=barrel_rate_against_delta,
        pa_against_current=pa_against_current,
        pa_against_prior=pa_against_prior,
    )


class TestSkillChangeConfig:
    def test_default_values(self) -> None:
        config = SkillChangeConfig()
        assert config.min_pa == 200
        assert config.barrel_rate_threshold == 0.02
        assert config.barrel_to_hr_factor == 0.5

    def test_custom_values(self) -> None:
        config = SkillChangeConfig(min_pa=300, barrel_rate_threshold=0.03)
        assert config.min_pa == 300
        assert config.barrel_rate_threshold == 0.03

    def test_frozen(self) -> None:
        config = SkillChangeConfig()
        with pytest.raises(AttributeError):
            config.min_pa = 300  # type: ignore[misc]


class TestSkillChangeAdjusterBatter:
    def test_no_adjustment_when_no_delta(self) -> None:
        computer = FakeDeltaComputer(batter_deltas={})
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        batter = make_batter()
        result = adjuster.adjust([batter])

        assert len(result) == 1
        assert result[0].rates["hr"] == 0.04  # unchanged

    def test_no_adjustment_when_insufficient_pa(self) -> None:
        delta = make_batter_delta(barrel_rate_delta=0.05, pa_current=100, pa_prior=150)
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        batter = make_batter()
        result = adjuster.adjust([batter])

        assert len(result) == 1
        assert result[0].rates["hr"] == 0.04  # unchanged
        assert "skill_change_adjustments" not in result[0].metadata

    def test_no_adjustment_when_delta_below_threshold(self) -> None:
        # barrel_rate_delta of 0.01 is below default threshold of 0.02
        delta = make_batter_delta(barrel_rate_delta=0.01)
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        batter = make_batter()
        result = adjuster.adjust([batter])

        assert len(result) == 1
        assert result[0].rates["hr"] == 0.04  # unchanged

    def test_barrel_rate_adjusts_hr_and_doubles(self) -> None:
        # +3% barrel rate (above 2% threshold)
        delta = make_batter_delta(barrel_rate_delta=0.03)
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        config = SkillChangeConfig(barrel_to_hr_factor=0.5, barrel_to_doubles_factor=0.3)
        adjuster = SkillChangeAdjuster(delta_computer=computer, config=config)

        batter = make_batter(rates={"hr": 0.04, "doubles": 0.05})
        result = adjuster.adjust([batter])

        assert len(result) == 1
        # HR: 0.04 + (0.03 * 0.5) = 0.04 + 0.015 = 0.055
        assert result[0].rates["hr"] == pytest.approx(0.055)
        # Doubles: 0.05 + (0.03 * 0.3) = 0.05 + 0.009 = 0.059
        assert result[0].rates["doubles"] == pytest.approx(0.059)
        assert "skill_change_adjustments" in result[0].metadata

    def test_exit_velo_adjusts_hr(self) -> None:
        # +2.0 mph exit velo (above 1.5 mph threshold)
        delta = make_batter_delta(exit_velo_max_delta=2.0)
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        config = SkillChangeConfig(exit_velo_to_hr_factor=0.005)
        adjuster = SkillChangeAdjuster(delta_computer=computer, config=config)

        batter = make_batter(rates={"hr": 0.04})
        result = adjuster.adjust([batter])

        # HR: 0.04 + (2.0 * 0.005) = 0.04 + 0.01 = 0.05
        assert result[0].rates["hr"] == pytest.approx(0.05)

    def test_chase_rate_adjusts_bb_and_so(self) -> None:
        # -4% chase rate (improved plate discipline)
        delta = make_batter_delta(chase_rate_delta=-0.04)
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        config = SkillChangeConfig(chase_to_bb_factor=-0.5, chase_to_so_factor=0.3)
        adjuster = SkillChangeAdjuster(delta_computer=computer, config=config)

        batter = make_batter(rates={"bb": 0.08, "so": 0.20})
        result = adjuster.adjust([batter])

        # BB: 0.08 + (-0.04 * -0.5) = 0.08 + 0.02 = 0.10
        assert result[0].rates["bb"] == pytest.approx(0.10)
        # SO: 0.20 + (-0.04 * 0.3) = 0.20 - 0.012 = 0.188
        assert result[0].rates["so"] == pytest.approx(0.188)

    def test_whiff_rate_adjusts_so(self) -> None:
        # +4% whiff rate (worse contact)
        delta = make_batter_delta(whiff_rate_delta=0.04)
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        config = SkillChangeConfig(whiff_to_so_factor=0.7)
        adjuster = SkillChangeAdjuster(delta_computer=computer, config=config)

        batter = make_batter(rates={"so": 0.20})
        result = adjuster.adjust([batter])

        # SO: 0.20 + (0.04 * 0.7) = 0.20 + 0.028 = 0.228
        assert result[0].rates["so"] == pytest.approx(0.228)

    def test_sprint_speed_adjusts_sb(self) -> None:
        # +1.0 ft/sec sprint speed
        delta = make_batter_delta(sprint_speed_delta=1.0)
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        config = SkillChangeConfig(sprint_to_sb_factor=0.02)
        adjuster = SkillChangeAdjuster(delta_computer=computer, config=config)

        batter = make_batter(rates={"sb": 0.02})
        result = adjuster.adjust([batter])

        # SB: 0.02 + (1.0 * 0.02) = 0.02 + 0.02 = 0.04
        assert result[0].rates["sb"] == pytest.approx(0.04)

    def test_sprint_speed_ignored_without_sb_rate(self) -> None:
        delta = make_batter_delta(sprint_speed_delta=1.0)
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        batter = make_batter(rates={"hr": 0.04})  # no sb rate
        result = adjuster.adjust([batter])

        assert "sb" not in result[0].rates

    def test_rates_dont_go_negative(self) -> None:
        # Large negative adjustment
        delta = make_batter_delta(barrel_rate_delta=-0.10)
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        config = SkillChangeConfig(barrel_to_hr_factor=0.5)
        adjuster = SkillChangeAdjuster(delta_computer=computer, config=config)

        batter = make_batter(rates={"hr": 0.02})  # small HR rate
        result = adjuster.adjust([batter])

        # Would be 0.02 + (-0.10 * 0.5) = 0.02 - 0.05 = -0.03, but clamped to 0
        assert result[0].rates["hr"] == 0.0

    def test_multiple_adjustments_combined(self) -> None:
        delta = make_batter_delta(
            barrel_rate_delta=0.03,
            chase_rate_delta=-0.04,
        )
        computer = FakeDeltaComputer(batter_deltas={"123": delta})
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        batter = make_batter(rates={"hr": 0.04, "doubles": 0.05, "bb": 0.08, "so": 0.20})
        result = adjuster.adjust([batter])

        # All adjustments should be applied
        assert result[0].rates["hr"] > 0.04
        assert result[0].rates["doubles"] > 0.05
        assert result[0].rates["bb"] > 0.08
        assert result[0].rates["so"] < 0.20


class TestSkillChangeAdjusterPitcher:
    def test_no_adjustment_when_no_delta(self) -> None:
        computer = FakeDeltaComputer(pitcher_deltas={})
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        pitcher = make_pitcher()
        result = adjuster.adjust([pitcher])

        assert len(result) == 1
        assert result[0].rates["so"] == 0.25  # unchanged

    def test_no_adjustment_when_insufficient_pa(self) -> None:
        delta = make_pitcher_delta(
            fastball_velo_delta=2.0,
            pa_against_current=100,
            pa_against_prior=150,
        )
        computer = FakeDeltaComputer(pitcher_deltas={"456": delta})
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        pitcher = make_pitcher()
        result = adjuster.adjust([pitcher])

        assert result[0].rates["so"] == 0.25  # unchanged

    def test_fastball_velo_adjusts_so_and_er(self) -> None:
        # +1.5 mph velocity
        delta = make_pitcher_delta(fastball_velo_delta=1.5)
        computer = FakeDeltaComputer(pitcher_deltas={"456": delta})
        config = SkillChangeConfig(velo_to_so_factor=0.003, velo_to_er_factor=-0.001)
        adjuster = SkillChangeAdjuster(delta_computer=computer, config=config)

        pitcher = make_pitcher(rates={"so": 0.25, "er": 0.10})
        result = adjuster.adjust([pitcher])

        # SO: 0.25 + (1.5 * 0.003) = 0.25 + 0.0045 = 0.2545
        assert result[0].rates["so"] == pytest.approx(0.2545)
        # ER: 0.10 + (1.5 * -0.001) = 0.10 - 0.0015 = 0.0985
        assert result[0].rates["er"] == pytest.approx(0.0985)

    def test_pitcher_whiff_rate_adjusts_so(self) -> None:
        # +4% whiff rate
        delta = make_pitcher_delta(whiff_rate_delta=0.04)
        computer = FakeDeltaComputer(pitcher_deltas={"456": delta})
        config = SkillChangeConfig(pitcher_whiff_to_so_factor=0.5)
        adjuster = SkillChangeAdjuster(delta_computer=computer, config=config)

        pitcher = make_pitcher(rates={"so": 0.25})
        result = adjuster.adjust([pitcher])

        # SO: 0.25 + (0.04 * 0.5) = 0.25 + 0.02 = 0.27
        assert result[0].rates["so"] == pytest.approx(0.27)

    def test_gb_rate_adjusts_hr(self) -> None:
        # +5% ground ball rate
        delta = make_pitcher_delta(gb_rate_delta=0.05)
        computer = FakeDeltaComputer(pitcher_deltas={"456": delta})
        config = SkillChangeConfig(gb_to_hr_factor=-0.1)
        adjuster = SkillChangeAdjuster(delta_computer=computer, config=config)

        pitcher = make_pitcher(rates={"hr": 0.03})
        result = adjuster.adjust([pitcher])

        # HR: 0.03 + (0.05 * -0.1) = 0.03 - 0.005 = 0.025
        assert result[0].rates["hr"] == pytest.approx(0.025)


class TestSkillChangeAdjusterMixed:
    def test_handles_mixed_batters_and_pitchers(self) -> None:
        batter_delta = make_batter_delta(player_id="123", barrel_rate_delta=0.03)
        pitcher_delta = make_pitcher_delta(player_id="456", fastball_velo_delta=1.5)
        computer = FakeDeltaComputer(
            batter_deltas={"123": batter_delta},
            pitcher_deltas={"456": pitcher_delta},
        )
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        batter = make_batter(player_id="123", rates={"hr": 0.04, "doubles": 0.05})
        pitcher = make_pitcher(player_id="456", rates={"so": 0.25, "er": 0.10, "hr": 0.03})

        result = adjuster.adjust([batter, pitcher])

        assert len(result) == 2
        # Batter adjusted (result[0] is batter)
        assert result[0].rates["hr"] > 0.04
        # Pitcher adjusted (result[1] is pitcher)
        assert result[1].rates["so"] > 0.25  # velo increased SO

    def test_empty_list_returns_empty(self) -> None:
        computer = FakeDeltaComputer()
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        result = adjuster.adjust([])

        assert result == []

    def test_caches_deltas_for_same_year(self) -> None:
        source = FakeSkillDataSource(
            batter_data={
                2023: [BatterSkillStats("123", "Test", 2023, 450, 0.16, 0.42, 92.0, 113.5, 0.28, 0.14, 28.5)],
                2024: [BatterSkillStats("123", "Test", 2024, 500, 0.18, 0.45, 93.5, 115.2, 0.25, 0.12, 29.5)],
            }
        )
        computer = SkillDeltaComputer(feature_store=make_test_feature_store(skill_data_source=source))
        adjuster = SkillChangeAdjuster(delta_computer=computer)

        batter1 = make_batter(player_id="123")
        batter2 = make_batter(player_id="123", name="Test Batter 2")

        # First call loads deltas
        adjuster.adjust([batter1])
        # Second call should use cached deltas
        adjuster.adjust([batter2])

        # Both should be adjusted (delta computer was called with 2025)
        # This test mainly verifies no errors occur with caching
