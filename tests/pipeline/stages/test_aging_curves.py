from fantasy_baseball_manager.marcel.league_averages import (
    BATTING_COMPONENT_STATS,
    PITCHING_COMPONENT_STATS,
)
from fantasy_baseball_manager.pipeline.stages.aging_curves import (
    BATTING_AGING_CURVES,
    PITCHING_AGING_CURVES,
    PITCHING_INVERTED_STATS,
    POSITION_AGING_MODIFIERS,
    AgingCurveParams,
)


class TestAgingCurveParams:
    def test_is_frozen(self) -> None:
        params = AgingCurveParams(peak_age=28, young_rate=0.006, old_rate=0.003)
        assert params.peak_age == 28
        assert params.young_rate == 0.006
        assert params.old_rate == 0.003


class TestBattingAgingCurves:
    def test_all_batting_stats_have_entries(self) -> None:
        for stat in BATTING_COMPONENT_STATS:
            assert stat in BATTING_AGING_CURVES, f"Missing aging curve for batting stat: {stat}"

    def test_all_values_positive(self) -> None:
        for stat, curve in BATTING_AGING_CURVES.items():
            assert curve.young_rate > 0, f"young_rate for {stat} must be positive"
            assert curve.old_rate > 0, f"old_rate for {stat} must be positive"

    def test_peak_ages_in_reasonable_range(self) -> None:
        for stat, curve in BATTING_AGING_CURVES.items():
            assert 24 <= curve.peak_age <= 32, f"peak_age for {stat} should be 24-32, got {curve.peak_age}"

    def test_power_peaks_later_than_speed(self) -> None:
        assert BATTING_AGING_CURVES["hr"].peak_age > BATTING_AGING_CURVES["sb"].peak_age

    def test_discipline_peaks_latest(self) -> None:
        discipline_peak = BATTING_AGING_CURVES["bb"].peak_age
        for stat in ("hr", "sb", "singles", "doubles"):
            assert discipline_peak >= BATTING_AGING_CURVES[stat].peak_age, f"Discipline should peak at or after {stat}"


class TestPitchingAgingCurves:
    def test_all_pitching_stats_have_entries(self) -> None:
        for stat in PITCHING_COMPONENT_STATS:
            assert stat in PITCHING_AGING_CURVES, f"Missing aging curve for pitching stat: {stat}"

    def test_all_values_positive(self) -> None:
        for stat, curve in PITCHING_AGING_CURVES.items():
            assert curve.young_rate > 0, f"young_rate for {stat} must be positive"
            assert curve.old_rate > 0, f"old_rate for {stat} must be positive"

    def test_peak_ages_in_reasonable_range(self) -> None:
        for stat, curve in PITCHING_AGING_CURVES.items():
            assert 24 <= curve.peak_age <= 32, f"peak_age for {stat} should be 24-32, got {curve.peak_age}"


class TestPitchingInvertedStats:
    def test_contains_expected_stats(self) -> None:
        for stat in ("bb", "hr", "h", "hbp", "er", "bs"):
            assert stat in PITCHING_INVERTED_STATS, f"{stat} should be in inverted stats"

    def test_excludes_so(self) -> None:
        assert "so" not in PITCHING_INVERTED_STATS

    def test_excludes_w(self) -> None:
        assert "w" not in PITCHING_INVERTED_STATS


class TestPositionAgingModifiers:
    def test_catcher_modifier_greater_than_one(self) -> None:
        assert POSITION_AGING_MODIFIERS["C"] > 1.0

    def test_infield_baseline(self) -> None:
        assert POSITION_AGING_MODIFIERS["IF"] == 1.0

    def test_outfield_baseline(self) -> None:
        assert POSITION_AGING_MODIFIERS["OF"] == 1.0

    def test_sp_baseline(self) -> None:
        assert POSITION_AGING_MODIFIERS["SP"] == 1.0
