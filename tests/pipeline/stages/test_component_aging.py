import pytest

from fantasy_baseball_manager.pipeline.stages.aging_curves import (
    BATTING_AGING_CURVES,
    AgingCurveParams,
)
from fantasy_baseball_manager.pipeline.stages.component_aging import (
    ComponentAgingAdjuster,
    component_age_multiplier,
)
from fantasy_baseball_manager.pipeline.types import PlayerMetadata, PlayerRates


class TestComponentAgeMultiplier:
    def test_at_peak_returns_one(self) -> None:
        curve = AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.015)
        assert component_age_multiplier(28, curve) == 1.0

    def test_young_boost(self) -> None:
        curve = AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.015)
        result = component_age_multiplier(25, curve)
        assert result == pytest.approx(1.0 + 3 * 0.006)

    def test_old_penalty(self) -> None:
        curve = AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.015)
        result = component_age_multiplier(32, curve)
        assert result == pytest.approx(1.0 - 4 * 0.015)

    def test_position_modifier_only_affects_decline(self) -> None:
        curve = AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.015)
        # Young: position modifier should not change result
        young_default = component_age_multiplier(25, curve, position_modifier=1.0)
        young_catcher = component_age_multiplier(25, curve, position_modifier=1.3)
        assert young_default == young_catcher

    def test_position_modifier_steepens_old_decline(self) -> None:
        curve = AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.015)
        old_default = component_age_multiplier(32, curve, position_modifier=1.0)
        old_catcher = component_age_multiplier(32, curve, position_modifier=1.3)
        assert old_catcher < old_default

    def test_exact_value_old_with_position_modifier(self) -> None:
        curve = AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.015)
        result = component_age_multiplier(32, curve, position_modifier=1.3)
        expected = 1.0 - 4 * 0.015 * 1.3
        assert result == pytest.approx(expected)


def _make_player(
    age: int,
    rates: dict[str, float],
    is_starter: bool | None = None,
    position: str | None = None,
) -> PlayerRates:
    metadata: PlayerMetadata = {}
    if is_starter is not None:
        metadata["is_starter"] = is_starter
    if position is not None:
        metadata["position"] = position
    return PlayerRates(
        player_id="p1",
        name="Test Player",
        year=2025,
        age=age,
        rates=rates,
        opportunities=600.0,
        metadata=metadata,
    )


class TestAdjusterBatting:
    def test_peak_age_no_change(self) -> None:
        # At HR peak age (28), HR should be unchanged
        player = _make_player(28, {"hr": 0.05}, is_starter=False)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        assert result[0].rates["hr"] == pytest.approx(0.05)

    def test_young_batter_boost(self) -> None:
        player = _make_player(25, {"hr": 0.05}, is_starter=False)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        expected_mult = component_age_multiplier(25, BATTING_AGING_CURVES["hr"])
        assert result[0].rates["hr"] == pytest.approx(0.05 * expected_mult)
        assert result[0].rates["hr"] > 0.05

    def test_old_batter_penalty(self) -> None:
        player = _make_player(35, {"hr": 0.05}, is_starter=False)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        assert result[0].rates["hr"] < 0.05

    def test_different_stats_get_different_multipliers(self) -> None:
        player = _make_player(22, {"hr": 0.05, "sb": 0.03, "bb": 0.08}, is_starter=False)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        hr_mult = result[0].rates["hr"] / 0.05
        sb_mult = result[0].rates["sb"] / 0.03
        bb_mult = result[0].rates["bb"] / 0.08
        # These should be different since they have different curves
        assert hr_mult != pytest.approx(sb_mult)
        assert hr_mult != pytest.approx(bb_mult)

    def test_speed_declines_faster_than_power(self) -> None:
        player = _make_player(34, {"hr": 0.05, "sb": 0.03}, is_starter=False)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        hr_mult = result[0].rates["hr"] / 0.05
        sb_mult = result[0].rates["sb"] / 0.03
        assert sb_mult < hr_mult


class TestAdjusterPitching:
    def test_detects_pitcher_via_is_starter(self) -> None:
        player = _make_player(26, {"so": 0.20}, is_starter=True)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        # At pitching SO peak (26), should be unchanged
        assert result[0].rates["so"] == pytest.approx(0.20)

    def test_young_pitcher_so_boosted(self) -> None:
        player = _make_player(23, {"so": 0.20}, is_starter=True)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        assert result[0].rates["so"] > 0.20

    def test_young_pitcher_bb_decreased_inverted(self) -> None:
        player = _make_player(23, {"bb": 0.08}, is_starter=True)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        # BB is inverted for pitchers: young pitcher gets 1/mult (decrease)
        assert result[0].rates["bb"] < 0.08

    def test_young_pitcher_hr_decreased_inverted(self) -> None:
        player = _make_player(23, {"hr": 0.03}, is_starter=True)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        assert result[0].rates["hr"] < 0.03

    def test_old_pitcher_bb_increased(self) -> None:
        player = _make_player(35, {"bb": 0.08}, is_starter=True)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        # BB is inverted: old pitcher penalty becomes boost (more walks)
        assert result[0].rates["bb"] > 0.08


class TestAdjusterPosition:
    def test_no_position_uses_default(self) -> None:
        player = _make_player(35, {"hr": 0.05}, is_starter=False)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        expected_mult = component_age_multiplier(35, BATTING_AGING_CURVES["hr"], position_modifier=1.0)
        assert result[0].rates["hr"] == pytest.approx(0.05 * expected_mult)

    def test_catcher_steeper_decline(self) -> None:
        player_if = _make_player(35, {"hr": 0.05}, is_starter=False, position="IF")
        player_c = _make_player(35, {"hr": 0.05}, is_starter=False, position="C")
        adjuster = ComponentAgingAdjuster()
        result_if = adjuster.adjust([player_if])
        result_c = adjuster.adjust([player_c])
        assert result_c[0].rates["hr"] < result_if[0].rates["hr"]

    def test_catcher_same_boost_when_young(self) -> None:
        player_if = _make_player(22, {"hr": 0.05}, is_starter=False, position="IF")
        player_c = _make_player(22, {"hr": 0.05}, is_starter=False, position="C")
        adjuster = ComponentAgingAdjuster()
        result_if = adjuster.adjust([player_if])
        result_c = adjuster.adjust([player_c])
        assert result_c[0].rates["hr"] == pytest.approx(result_if[0].rates["hr"])


class TestAdjusterDependencyInjection:
    def test_custom_curves_are_used(self) -> None:
        custom_batting = {
            "hr": AgingCurveParams(peak_age=30, young_rate=0.010, old_rate=0.020),
        }
        player = _make_player(25, {"hr": 0.05}, is_starter=False)
        adjuster = ComponentAgingAdjuster(batting_curves=custom_batting)
        result = adjuster.adjust([player])
        expected_mult = component_age_multiplier(25, custom_batting["hr"])
        assert result[0].rates["hr"] == pytest.approx(0.05 * expected_mult)

    def test_custom_position_modifiers(self) -> None:
        custom_mods = {"C": 2.0}
        player = _make_player(35, {"hr": 0.05}, is_starter=False, position="C")
        adjuster = ComponentAgingAdjuster(position_modifiers=custom_mods)
        result = adjuster.adjust([player])
        expected_mult = component_age_multiplier(35, BATTING_AGING_CURVES["hr"], position_modifier=2.0)
        assert result[0].rates["hr"] == pytest.approx(0.05 * expected_mult)


class TestAdjusterEdgeCases:
    def test_empty_input(self) -> None:
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([])
        assert result == []

    def test_unknown_stat_passthrough(self) -> None:
        player = _make_player(25, {"xyz": 0.10}, is_starter=False)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        assert result[0].rates["xyz"] == 0.10

    def test_metadata_preserved(self) -> None:
        player = _make_player(25, {"hr": 0.05}, is_starter=False, position="C")
        player.metadata["custom_key"] = "custom_value"  # type: ignore[typeddict-unknown-key]
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        assert result[0].metadata["custom_key"] == "custom_value"  # type: ignore[typeddict-item]
        assert result[0].metadata["is_starter"] is False
        assert result[0].metadata["position"] == "C"

    def test_opportunities_preserved(self) -> None:
        player = _make_player(25, {"hr": 0.05}, is_starter=False)
        adjuster = ComponentAgingAdjuster()
        result = adjuster.adjust([player])
        assert result[0].opportunities == 600.0
