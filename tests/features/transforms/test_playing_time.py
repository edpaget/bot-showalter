from __future__ import annotations

from fantasy_baseball_manager.features.transforms.playing_time import (
    make_il_severity_transform,
    make_il_summary_transform,
    make_pt_interaction_transform,
    make_pt_trend_transform,
    make_starter_ratio_transform,
    make_war_threshold_transform,
)


class TestStarterRatioTransform:
    def test_full_starter(self) -> None:
        transform = make_starter_ratio_transform()
        result = transform([{"gs_1": 30, "g_1": 30}])
        assert result == {"starter_ratio": 1.0}

    def test_pure_reliever(self) -> None:
        transform = make_starter_ratio_transform()
        result = transform([{"gs_1": 0, "g_1": 60}])
        assert result == {"starter_ratio": 0.0}

    def test_swingman(self) -> None:
        transform = make_starter_ratio_transform()
        result = transform([{"gs_1": 15, "g_1": 45}])
        assert result["starter_ratio"] == 15 / 45

    def test_zero_games_returns_zero(self) -> None:
        transform = make_starter_ratio_transform()
        result = transform([{"gs_1": 0, "g_1": 0}])
        assert result == {"starter_ratio": 0.0}

    def test_none_g_returns_zero(self) -> None:
        transform = make_starter_ratio_transform()
        result = transform([{"gs_1": 10, "g_1": None}])
        assert result == {"starter_ratio": 0.0}

    def test_none_gs_returns_zero(self) -> None:
        transform = make_starter_ratio_transform()
        result = transform([{"gs_1": None, "g_1": 30}])
        assert result == {"starter_ratio": 0.0}

    def test_both_none_returns_zero(self) -> None:
        transform = make_starter_ratio_transform()
        result = transform([{"gs_1": None, "g_1": None}])
        assert result == {"starter_ratio": 0.0}

    def test_missing_keys_returns_zero(self) -> None:
        transform = make_starter_ratio_transform()
        result = transform([{}])
        assert result == {"starter_ratio": 0.0}


class TestIlSeverityTransform:
    def test_minor_il_stint(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{"il_days_1": 15}])
        assert result == {"il_minor": 1.0, "il_moderate": 0.0, "il_severe": 0.0}

    def test_moderate_il_stint(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{"il_days_1": 45}])
        assert result == {"il_minor": 0.0, "il_moderate": 1.0, "il_severe": 0.0}

    def test_severe_il_stint(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{"il_days_1": 90}])
        assert result == {"il_minor": 0.0, "il_moderate": 0.0, "il_severe": 1.0}

    def test_zero_days_sets_no_flags(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{"il_days_1": 0}])
        assert result == {"il_minor": 0.0, "il_moderate": 0.0, "il_severe": 0.0}

    def test_boundary_at_20(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{"il_days_1": 20}])
        assert result["il_minor"] == 1.0

    def test_boundary_at_21(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{"il_days_1": 21}])
        assert result["il_minor"] == 0.0
        assert result["il_moderate"] == 1.0

    def test_boundary_at_60(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{"il_days_1": 60}])
        assert result["il_moderate"] == 1.0

    def test_boundary_at_61(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{"il_days_1": 61}])
        assert result["il_moderate"] == 0.0
        assert result["il_severe"] == 1.0

    def test_none_treated_as_zero(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{"il_days_1": None}])
        assert result == {"il_minor": 0.0, "il_moderate": 0.0, "il_severe": 0.0}

    def test_missing_key_treated_as_zero(self) -> None:
        transform = make_il_severity_transform()
        result = transform([{}])
        assert result == {"il_minor": 0.0, "il_moderate": 0.0, "il_severe": 0.0}


class TestILSummaryTransform:
    def test_il_days_3yr_sums_all_values(self) -> None:
        transform = make_il_summary_transform(lags=3)
        row = {
            "il_days_1": 30,
            "il_days_2": 15,
            "il_days_3": 10,
            "il_stints_1": 2,
            "il_stints_2": 1,
        }
        result = transform([row])
        assert result["il_days_3yr"] == 55

    def test_il_days_3yr_treats_none_as_zero(self) -> None:
        transform = make_il_summary_transform(lags=3)
        row = {
            "il_days_1": 30,
            "il_days_2": None,
            "il_days_3": None,
            "il_stints_1": 2,
            "il_stints_2": 0,
        }
        result = transform([row])
        assert result["il_days_3yr"] == 30

    def test_il_recurrence_both_seasons_have_stints(self) -> None:
        transform = make_il_summary_transform(lags=3)
        row = {
            "il_days_1": 30,
            "il_days_2": 15,
            "il_days_3": 0,
            "il_stints_1": 2,
            "il_stints_2": 1,
        }
        result = transform([row])
        assert result["il_recurrence"] == 1.0

    def test_il_recurrence_only_one_season(self) -> None:
        transform = make_il_summary_transform(lags=3)
        row = {
            "il_days_1": 30,
            "il_days_2": 0,
            "il_days_3": 0,
            "il_stints_1": 2,
            "il_stints_2": 0,
        }
        result = transform([row])
        assert result["il_recurrence"] == 0.0

    def test_il_recurrence_no_stints(self) -> None:
        transform = make_il_summary_transform(lags=3)
        row = {
            "il_days_1": 0,
            "il_days_2": 0,
            "il_days_3": 0,
            "il_stints_1": 0,
            "il_stints_2": 0,
        }
        result = transform([row])
        assert result["il_recurrence"] == 0.0

    def test_il_recurrence_none_stints_treated_as_zero(self) -> None:
        transform = make_il_summary_transform(lags=3)
        row = {
            "il_days_1": None,
            "il_days_2": None,
            "il_days_3": None,
            "il_stints_1": None,
            "il_stints_2": None,
        }
        result = transform([row])
        assert result["il_recurrence"] == 0.0


class TestWarThresholdTransform:
    def test_high_war_sets_both_flags(self) -> None:
        transform = make_war_threshold_transform()
        result = transform([{"war_1": 5.0}])
        assert result == {"war_above_2": 1.0, "war_above_4": 1.0, "war_below_0": 0.0}

    def test_moderate_war_sets_above_2_only(self) -> None:
        transform = make_war_threshold_transform()
        result = transform([{"war_1": 3.0}])
        assert result == {"war_above_2": 1.0, "war_above_4": 0.0, "war_below_0": 0.0}

    def test_low_positive_war_sets_no_flags(self) -> None:
        transform = make_war_threshold_transform()
        result = transform([{"war_1": 1.0}])
        assert result == {"war_above_2": 0.0, "war_above_4": 0.0, "war_below_0": 0.0}

    def test_negative_war_sets_below_0(self) -> None:
        transform = make_war_threshold_transform()
        result = transform([{"war_1": -0.5}])
        assert result == {"war_above_2": 0.0, "war_above_4": 0.0, "war_below_0": 1.0}

    def test_boundary_at_2(self) -> None:
        transform = make_war_threshold_transform()
        result = transform([{"war_1": 2.0}])
        assert result["war_above_2"] == 1.0

    def test_boundary_at_4(self) -> None:
        transform = make_war_threshold_transform()
        result = transform([{"war_1": 4.0}])
        assert result["war_above_4"] == 1.0

    def test_boundary_at_0(self) -> None:
        transform = make_war_threshold_transform()
        result = transform([{"war_1": 0.0}])
        assert result["war_below_0"] == 0.0

    def test_none_war_treated_as_zero(self) -> None:
        transform = make_war_threshold_transform()
        result = transform([{"war_1": None}])
        assert result == {"war_above_2": 0.0, "war_above_4": 0.0, "war_below_0": 0.0}

    def test_missing_key_treated_as_zero(self) -> None:
        transform = make_war_threshold_transform()
        result = transform([{}])
        assert result == {"war_above_2": 0.0, "war_above_4": 0.0, "war_below_0": 0.0}


class TestPtTrendTransform:
    def test_computes_correct_ratio(self) -> None:
        transform = make_pt_trend_transform("pa")
        row = {"pa_1": 500, "pa_2": 400}
        result = transform([row])
        assert result["pt_trend"] == 500 / 400

    def test_returns_1_when_denominator_is_zero(self) -> None:
        transform = make_pt_trend_transform("pa")
        row = {"pa_1": 500, "pa_2": 0}
        result = transform([row])
        assert result["pt_trend"] == 1.0

    def test_returns_1_when_denominator_is_none(self) -> None:
        transform = make_pt_trend_transform("pa")
        row = {"pa_1": 500, "pa_2": None}
        result = transform([row])
        assert result["pt_trend"] == 1.0

    def test_returns_1_when_numerator_is_none(self) -> None:
        transform = make_pt_trend_transform("pa")
        row = {"pa_1": None, "pa_2": 400}
        result = transform([row])
        assert result["pt_trend"] == 1.0

    def test_ip_column(self) -> None:
        transform = make_pt_trend_transform("ip")
        row = {"ip_1": 180.0, "ip_2": 160.0}
        result = transform([row])
        assert result["pt_trend"] == 180.0 / 160.0


class TestPtInteractionTransform:
    def test_war_trend_product(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": 3.0, "pt_trend": 1.2, "age": 25, "il_recurrence": 0.0}])
        assert result["war_trend"] == 3.0 * 1.2

    def test_age_il_older_with_recurrence(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": 2.0, "pt_trend": 1.0, "age": 35, "il_recurrence": 1.0}])
        assert result["age_il_interact"] == 5.0 * 1.0

    def test_age_il_young_with_recurrence(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": 2.0, "pt_trend": 1.0, "age": 25, "il_recurrence": 1.0}])
        assert result["age_il_interact"] == 0.0

    def test_age_il_older_no_recurrence(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": 2.0, "pt_trend": 1.0, "age": 35, "il_recurrence": 0.0}])
        assert result["age_il_interact"] == 0.0

    def test_age_boundary_at_30(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": 1.0, "pt_trend": 1.0, "age": 30, "il_recurrence": 1.0}])
        assert result["age_il_interact"] == 0.0

    def test_age_boundary_at_31(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": 1.0, "pt_trend": 1.0, "age": 31, "il_recurrence": 1.0}])
        assert result["age_il_interact"] == 1.0

    def test_none_war(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": None, "pt_trend": 1.5, "age": 25, "il_recurrence": 0.0}])
        assert result["war_trend"] == 0.0

    def test_none_pt_trend_defaults_to_1(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": 3.0, "pt_trend": None, "age": 25, "il_recurrence": 0.0}])
        assert result["war_trend"] == 3.0

    def test_none_age(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": 2.0, "pt_trend": 1.0, "age": None, "il_recurrence": 1.0}])
        assert result["age_il_interact"] == 0.0

    def test_none_il_recurrence(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{"war_1": 2.0, "pt_trend": 1.0, "age": 35, "il_recurrence": None}])
        assert result["age_il_interact"] == 0.0

    def test_all_missing(self) -> None:
        transform = make_pt_interaction_transform()
        result = transform([{}])
        assert result == {"war_trend": 0.0, "age_il_interact": 0.0}
