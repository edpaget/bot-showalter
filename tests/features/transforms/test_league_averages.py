from __future__ import annotations

import pytest

from fantasy_baseball_manager.features.transforms.league_averages import (
    make_league_avg_transform,
)


class TestMakeLeagueAvgTransform:
    """Test the factory-generated league-average transform."""

    def test_two_players_batting(self) -> None:
        transform = make_league_avg_transform(
            categories=("hr", "h"),
            pt_column="pa",
        )
        rows = [
            {"hr_1": 30.0, "h_1": 150.0, "pa_1": 600},
            {"hr_1": 20.0, "h_1": 130.0, "pa_1": 500},
        ]
        result = transform(rows)
        # total hr = 30 + 20 = 50, total PA = 600 + 500 = 1100
        assert result["league_hr_rate"] == pytest.approx(50.0 / 1100.0)
        assert result["league_h_rate"] == pytest.approx(280.0 / 1100.0)

    def test_single_player(self) -> None:
        transform = make_league_avg_transform(
            categories=("hr",),
            pt_column="pa",
        )
        rows = [
            {"hr_1": 30.0, "pa_1": 600},
        ]
        result = transform(rows)
        assert result["league_hr_rate"] == pytest.approx(30.0 / 600.0)

    def test_pitcher_with_ip(self) -> None:
        transform = make_league_avg_transform(
            categories=("so",),
            pt_column="ip",
        )
        rows = [
            {"so_1": 200.0, "ip_1": 180.0},
            {"so_1": 150.0, "ip_1": 170.0},
        ]
        result = transform(rows)
        # so: (200 + 150) / (180 + 170) = 350/350
        assert result["league_so_rate"] == pytest.approx(350.0 / 350.0)

    def test_zero_pt_player_excluded(self) -> None:
        transform = make_league_avg_transform(
            categories=("hr",),
            pt_column="pa",
        )
        rows = [
            {"hr_1": 30.0, "pa_1": 600},
            {"hr_1": 5.0, "pa_1": 0},  # zero PT — contributes nothing
        ]
        result = transform(rows)
        # Only player 1 contributes: 30/600
        assert result["league_hr_rate"] == pytest.approx(30.0 / 600.0)

    def test_null_pt_player_excluded(self) -> None:
        transform = make_league_avg_transform(
            categories=("hr",),
            pt_column="pa",
        )
        rows = [
            {"hr_1": 30.0, "pa_1": 600},
            {"hr_1": None, "pa_1": None},
        ]
        result = transform(rows)
        assert result["league_hr_rate"] == pytest.approx(30.0 / 600.0)

    def test_all_zero_pt_returns_zeros(self) -> None:
        transform = make_league_avg_transform(
            categories=("hr",),
            pt_column="pa",
        )
        rows = [
            {"hr_1": 0.0, "pa_1": 0},
            {"hr_1": 0.0, "pa_1": 0},
        ]
        result = transform(rows)
        assert result["league_hr_rate"] == 0.0

    def test_empty_rows_returns_zeros(self) -> None:
        transform = make_league_avg_transform(
            categories=("hr", "h"),
            pt_column="pa",
        )
        result = transform([])
        assert result["league_hr_rate"] == 0.0
        assert result["league_h_rate"] == 0.0
