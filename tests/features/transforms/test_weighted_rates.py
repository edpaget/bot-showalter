from __future__ import annotations

import pytest

from fantasy_baseball_manager.features.transforms.weighted_rates import (
    make_weighted_rates_transform,
)


class TestMakeWeightedRatesTransform:
    """Test the factory-generated transform against known inputs."""

    def test_three_year_batter(self) -> None:
        transform = make_weighted_rates_transform(
            categories=("hr", "h"),
            weights=(5.0, 4.0, 3.0),
            pt_column="pa",
        )
        rows = [
            {
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": 20.0,
                "h_1": 150.0,
                "h_2": 140.0,
                "h_3": 120.0,
                "pa_1": 600,
                "pa_2": 500,
                "pa_3": 400,
            },
        ]
        result = transform(rows)
        # hr: (30*5 + 25*4 + 20*3) / (600*5 + 500*4 + 400*3)
        # = 310 / 6200
        assert result["hr_wavg"] == pytest.approx(310.0 / 6200.0)
        # h: (150*5 + 140*4 + 120*3) / 6200 = 1670/6200
        assert result["h_wavg"] == pytest.approx(1670.0 / 6200.0)
        # weighted_pt: 6200
        assert result["weighted_pt"] == pytest.approx(6200.0)

    def test_two_year_batter_with_three_weights(self) -> None:
        transform = make_weighted_rates_transform(
            categories=("hr",),
            weights=(5.0, 4.0, 3.0),
            pt_column="pa",
        )
        rows = [
            {
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_3": None,
                "pa_1": 600,
                "pa_2": 500,
                "pa_3": None,
            },
        ]
        result = transform(rows)
        # Only 2 non-null seasons
        # hr: (30*5 + 25*4) / (600*5 + 500*4) = 250/5000
        assert result["hr_wavg"] == pytest.approx(250.0 / 5000.0)
        assert result["weighted_pt"] == pytest.approx(5000.0)

    def test_one_year_batter(self) -> None:
        transform = make_weighted_rates_transform(
            categories=("hr",),
            weights=(5.0, 4.0, 3.0),
            pt_column="pa",
        )
        rows = [
            {
                "hr_1": 30.0,
                "hr_2": None,
                "hr_3": None,
                "pa_1": 600,
                "pa_2": None,
                "pa_3": None,
            },
        ]
        result = transform(rows)
        # hr: 30*5 / (600*5) = 30/600
        assert result["hr_wavg"] == pytest.approx(30.0 / 600.0)
        assert result["weighted_pt"] == pytest.approx(3000.0)

    def test_zero_pt_season_contributes_nothing(self) -> None:
        transform = make_weighted_rates_transform(
            categories=("hr",),
            weights=(5.0, 4.0, 3.0),
            pt_column="pa",
        )
        rows = [
            {
                "hr_1": 30.0,
                "hr_2": 0.0,
                "hr_3": 20.0,
                "pa_1": 600,
                "pa_2": 0,
                "pa_3": 400,
            },
        ]
        result = transform(rows)
        # 0-PA season contributes 0 weighted stats and 0 weighted PA
        # hr: (30*5 + 0*4 + 20*3) / (600*5 + 0*4 + 400*3) = 210/4200
        assert result["hr_wavg"] == pytest.approx(210.0 / 4200.0)
        assert result["weighted_pt"] == pytest.approx(4200.0)

    def test_pitcher_with_ip(self) -> None:
        transform = make_weighted_rates_transform(
            categories=("so", "er"),
            weights=(3.0, 2.0, 1.0),
            pt_column="ip",
        )
        rows = [
            {
                "so_1": 200.0,
                "so_2": 180.0,
                "so_3": 150.0,
                "er_1": 60.0,
                "er_2": 55.0,
                "er_3": 50.0,
                "ip_1": 180.0,
                "ip_2": 170.0,
                "ip_3": 160.0,
            },
        ]
        result = transform(rows)
        # so: (200*3 + 180*2 + 150*1) / (180*3 + 170*2 + 160*1) = 1110/1040
        assert result["so_wavg"] == pytest.approx(1110.0 / 1040.0)
        assert result["weighted_pt"] == pytest.approx(1040.0)

    def test_all_null_returns_zeros(self) -> None:
        transform = make_weighted_rates_transform(
            categories=("hr",),
            weights=(5.0, 4.0, 3.0),
            pt_column="pa",
        )
        rows = [
            {
                "hr_1": None,
                "hr_2": None,
                "hr_3": None,
                "pa_1": None,
                "pa_2": None,
                "pa_3": None,
            },
        ]
        result = transform(rows)
        assert result["hr_wavg"] == 0.0
        assert result["weighted_pt"] == 0.0

    def test_all_zero_pt_returns_zeros(self) -> None:
        transform = make_weighted_rates_transform(
            categories=("hr",),
            weights=(5.0, 4.0, 3.0),
            pt_column="pa",
        )
        rows = [
            {
                "hr_1": 0.0,
                "hr_2": 0.0,
                "hr_3": 0.0,
                "pa_1": 0,
                "pa_2": 0,
                "pa_3": 0,
            },
        ]
        result = transform(rows)
        assert result["hr_wavg"] == 0.0
        assert result["weighted_pt"] == 0.0
