from __future__ import annotations

from fantasy_baseball_manager.features.transforms.playing_time import (
    make_il_summary_transform,
    make_pt_trend_transform,
)


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

    def test_ip_column(self) -> None:
        transform = make_pt_trend_transform("ip")
        row = {"ip_1": 180.0, "ip_2": 160.0}
        result = transform([row])
        assert result["pt_trend"] == 180.0 / 160.0
