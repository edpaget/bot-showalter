from fantasy_baseball_manager.services.quick_eval import QuickEvalResult, quick_eval


def _make_rows(n: int, seasons: list[int]) -> dict[int, list[dict[str, float | int]]]:
    """Build synthetic rows where target_slg = feature_a * 0.5 + feature_b * 0.3."""
    rows_by_season: dict[int, list[dict[str, float | int]]] = {}
    i = 0
    for season in seasons:
        season_rows: list[dict[str, float | int]] = []
        for _ in range(n):
            a = (i % 10) * 0.1
            b = ((i + 3) % 10) * 0.1
            season_rows.append(
                {
                    "season": season,
                    "feature_a": a,
                    "feature_b": b,
                    "target_slg": a * 0.5 + b * 0.3,
                    "target_avg": a * 0.3,
                    "target_obp": a * 0.4,
                }
            )
            i += 1
        rows_by_season[season] = season_rows
    return rows_by_season


class TestQuickEvalCorrectRMSE:
    def test_rmse_near_zero_for_learnable_relationship(self) -> None:
        rows = _make_rows(50, [2022, 2023, 2024])
        result = quick_eval(
            feature_columns=["feature_a", "feature_b"],
            target="slg",
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
        )
        assert result.rmse < 0.05
        assert isinstance(result.rmse, float)


class TestQuickEvalSingleTargetIsolation:
    def test_only_specified_target_is_trained(self) -> None:
        rows = _make_rows(50, [2022, 2023, 2024])
        result = quick_eval(
            feature_columns=["feature_a", "feature_b"],
            target="slg",
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
        )
        assert result.target == "slg"

    def test_other_targets_in_data_are_ignored(self) -> None:
        rows = _make_rows(50, [2022, 2023, 2024])
        result = quick_eval(
            feature_columns=["feature_a", "feature_b"],
            target="avg",
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
        )
        assert result.target == "avg"
        assert result.rmse < 0.05


class TestQuickEvalDeltaWithBaseline:
    def test_delta_computed_when_baseline_provided(self) -> None:
        rows = _make_rows(50, [2022, 2023, 2024])
        result = quick_eval(
            feature_columns=["feature_a", "feature_b"],
            target="slg",
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
            baseline_rmse=0.1,
        )
        assert result.baseline_rmse == 0.1
        assert result.delta is not None
        assert result.delta == result.rmse - 0.1
        assert result.delta_pct is not None
        assert abs(result.delta_pct - (result.delta / 0.1 * 100)) < 1e-10


class TestQuickEvalDeltaWithoutBaseline:
    def test_delta_none_when_no_baseline(self) -> None:
        rows = _make_rows(50, [2022, 2023, 2024])
        result = quick_eval(
            feature_columns=["feature_a", "feature_b"],
            target="slg",
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
        )
        assert result.baseline_rmse is None
        assert result.delta is None
        assert result.delta_pct is None


class TestQuickEvalRSquaredAndN:
    def test_r_squared_and_n_populated(self) -> None:
        rows = _make_rows(50, [2022, 2023, 2024])
        result = quick_eval(
            feature_columns=["feature_a", "feature_b"],
            target="slg",
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
        )
        assert result.r_squared > 0.5
        assert result.n == 50

    def test_result_is_frozen_dataclass(self) -> None:
        result = QuickEvalResult(target="slg", rmse=0.05, r_squared=0.9, n=100)
        assert result.target == "slg"
        assert result.rmse == 0.05
        assert result.r_squared == 0.9
        assert result.n == 100


class TestQuickEvalCustomParams:
    def test_custom_params_passed_through(self) -> None:
        rows = _make_rows(50, [2022, 2023, 2024])
        result = quick_eval(
            feature_columns=["feature_a", "feature_b"],
            target="slg",
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
            params={"max_iter": 50, "max_depth": 3},
        )
        assert result.rmse >= 0.0
