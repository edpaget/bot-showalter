from typing import Any

import numpy as np
import pytest

from fantasy_baseball_manager.models.sampling import (
    holdout_metrics,
    season_kfold,
    temporal_expanding_cv,
    temporal_holdout_split,
)


def _make_row(player_id: int, season: int, season_column: str = "season") -> dict[str, Any]:
    """Create a minimal row with a player id and season."""
    return {"player_id": player_id, season_column: season}


def _make_rows(
    seasons: list[int],
    players_per_season: int = 3,
    season_column: str = "season",
) -> list[dict[str, Any]]:
    """Create rows across multiple seasons."""
    return [_make_row(p, s, season_column) for s in seasons for p in range(players_per_season)]


class TestHoldoutMetrics:
    def test_perfect_predictions(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        result = holdout_metrics(y, y)

        assert result["r_squared"] == pytest.approx(1.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["n"] == 3

    def test_known_values(self) -> None:
        y_actual = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])

        ss_res = 0.5**2 + 0.5**2 + 0.0**2 + 1.0**2  # 1.5
        mean_actual = (3.0 + -0.5 + 2.0 + 7.0) / 4  # 2.875
        ss_tot = sum((v - mean_actual) ** 2 for v in [3.0, -0.5, 2.0, 7.0])
        expected_r2 = 1.0 - ss_res / ss_tot
        expected_rmse = (ss_res / 4) ** 0.5

        result = holdout_metrics(y_actual, y_pred)

        assert result["r_squared"] == pytest.approx(expected_r2)
        assert result["rmse"] == pytest.approx(expected_rmse)
        assert result["n"] == 4

    def test_empty_arrays(self) -> None:
        result = holdout_metrics(np.array([]), np.array([]))

        assert result["r_squared"] == 0.0
        assert result["rmse"] == 0.0
        assert result["n"] == 0


class TestTemporalHoldoutSplit:
    def test_holdout_contains_only_max_season(self) -> None:
        rows = _make_rows([2020, 2021, 2022])
        train, holdout = temporal_holdout_split(rows)

        assert all(r["season"] == 2022 for r in holdout)
        assert all(r["season"] in (2020, 2021) for r in train)

    def test_partition_is_exhaustive(self) -> None:
        rows = _make_rows([2020, 2021, 2022])
        train, holdout = temporal_holdout_split(rows)

        assert len(train) + len(holdout) == len(rows)

    def test_raises_on_single_season(self) -> None:
        rows = _make_rows([2022])
        with pytest.raises(ValueError, match="at least 2"):
            temporal_holdout_split(rows)

    def test_raises_on_empty_rows(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            temporal_holdout_split([])

    def test_custom_season_column(self) -> None:
        rows = _make_rows([2021, 2022], season_column="year")
        train, holdout = temporal_holdout_split(rows, season_column="year")

        assert all(r["year"] == 2022 for r in holdout)
        assert all(r["year"] == 2021 for r in train)


class TestTemporalExpandingCV:
    def test_yields_correct_number_of_folds(self) -> None:
        # [2020, 2021, 2022, 2023] -> 2 folds (last season reserved for holdout)
        folds = list(temporal_expanding_cv([2020, 2021, 2022, 2023]))
        assert len(folds) == 2

    def test_train_always_before_test(self) -> None:
        folds = list(temporal_expanding_cv([2020, 2021, 2022, 2023]))
        for train_seasons, test_season in folds:
            assert all(s < test_season for s in train_seasons)

    def test_expanding_train_window(self) -> None:
        folds = list(temporal_expanding_cv([2020, 2021, 2022, 2023]))
        # Fold 0: train=[2020], test=2021
        assert folds[0] == ([2020], 2021)
        # Fold 1: train=[2020, 2021], test=2022
        assert folds[1] == ([2020, 2021], 2022)

    def test_raises_on_fewer_than_3_seasons(self) -> None:
        with pytest.raises(ValueError, match="at least 3 seasons"):
            list(temporal_expanding_cv([2022, 2023]))

    def test_raises_on_single_season(self) -> None:
        with pytest.raises(ValueError, match="at least 3 seasons"):
            list(temporal_expanding_cv([2022]))

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="at least 3 seasons"):
            list(temporal_expanding_cv([]))

    def test_three_seasons_yields_one_fold(self) -> None:
        folds = list(temporal_expanding_cv([2021, 2022, 2023]))
        assert len(folds) == 1
        assert folds[0] == ([2021], 2022)

    def test_unsorted_input_still_respects_temporal_order(self) -> None:
        folds = list(temporal_expanding_cv([2023, 2020, 2022, 2021]))
        assert folds[0] == ([2020], 2021)
        assert folds[1] == ([2020, 2021], 2022)


class TestSeasonKFold:
    def test_yields_correct_number_of_folds(self) -> None:
        rows = _make_rows([2020, 2021, 2022, 2023])
        folds = list(season_kfold(rows, n_folds=4))

        assert len(folds) == 4

    def test_folds_capped_by_season_count(self) -> None:
        rows = _make_rows([2020, 2021, 2022])
        folds = list(season_kfold(rows, n_folds=5))

        assert len(folds) == 3

    def test_no_season_in_both_train_and_test(self) -> None:
        rows = _make_rows([2020, 2021, 2022, 2023])
        for train, test in season_kfold(rows, n_folds=4):
            train_seasons = {r["season"] for r in train}
            test_seasons = {r["season"] for r in test}
            assert train_seasons.isdisjoint(test_seasons)

    def test_partition_is_exhaustive_per_fold(self) -> None:
        rows = _make_rows([2020, 2021, 2022, 2023])
        for train, test in season_kfold(rows, n_folds=4):
            assert len(train) + len(test) == len(rows)

    def test_single_season_yields_nothing(self) -> None:
        rows = _make_rows([2022])
        folds = list(season_kfold(rows))

        assert folds == []

    def test_custom_season_column(self) -> None:
        rows = _make_rows([2021, 2022], season_column="year")
        folds = list(season_kfold(rows, season_column="year"))

        assert len(folds) > 0
        for train, test in folds:
            train_seasons = {r["year"] for r in train}
            test_seasons = {r["year"] for r in test}
            assert train_seasons.isdisjoint(test_seasons)
