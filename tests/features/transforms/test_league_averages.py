from __future__ import annotations

import pytest

from fantasy_baseball_manager.features.transforms.league_averages import (
    make_league_avg_transform,
)
from fantasy_baseball_manager.models.marcel.engine import compute_league_averages
from fantasy_baseball_manager.models.marcel.types import SeasonLine


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

    def test_matches_engine_compute_league_averages_batting(self) -> None:
        """Verify the transform matches the existing engine function for batters."""
        categories = ("hr", "h", "bb")

        # Engine format: dict[player_id, list[SeasonLine]]
        all_seasons = {
            1: [SeasonLine(stats={"hr": 30.0, "h": 150.0, "bb": 60.0}, pa=600)],
            2: [SeasonLine(stats={"hr": 20.0, "h": 130.0, "bb": 50.0}, pa=500)],
            3: [SeasonLine(stats={"hr": 25.0, "h": 140.0, "bb": 55.0}, pa=550)],
        }
        expected = compute_league_averages(all_seasons, categories)

        # Transform format: list of rows with lag-1 columns
        transform = make_league_avg_transform(categories=categories, pt_column="pa")
        rows = [
            {"hr_1": 30.0, "h_1": 150.0, "bb_1": 60.0, "pa_1": 600},
            {"hr_1": 20.0, "h_1": 130.0, "bb_1": 50.0, "pa_1": 500},
            {"hr_1": 25.0, "h_1": 140.0, "bb_1": 55.0, "pa_1": 550},
        ]
        result = transform(rows)
        for cat in categories:
            assert result[f"league_{cat}_rate"] == pytest.approx(expected.rates[cat])

    def test_matches_engine_compute_league_averages_pitching(self) -> None:
        """Verify the transform matches the existing engine function for pitchers."""
        categories = ("so", "er")

        all_seasons = {
            1: [SeasonLine(stats={"so": 200.0, "er": 60.0}, ip=180.0, g=30, gs=30)],
            2: [SeasonLine(stats={"so": 150.0, "er": 50.0}, ip=170.0, g=28, gs=28)],
        }
        expected = compute_league_averages(all_seasons, categories)

        transform = make_league_avg_transform(categories=categories, pt_column="ip")
        rows = [
            {"so_1": 200.0, "er_1": 60.0, "ip_1": 180.0},
            {"so_1": 150.0, "er_1": 50.0, "ip_1": 170.0},
        ]
        result = transform(rows)
        for cat in categories:
            assert result[f"league_{cat}_rate"] == pytest.approx(expected.rates[cat])
