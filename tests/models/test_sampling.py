from typing import Any

import pytest

from fantasy_baseball_manager.models.sampling import (
    season_kfold,
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
