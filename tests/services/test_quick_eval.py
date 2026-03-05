import pytest

from fantasy_baseball_manager.models.gbm_training_backend import GBMTrainingBackend
from fantasy_baseball_manager.services.quick_eval import (
    FeatureSetComparisonResult,
    QuickEvalResult,
    compare_feature_sets,
    marginal_value,
    quick_eval,
)

_backend = GBMTrainingBackend()


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
            backend=_backend,
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
            backend=_backend,
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
            backend=_backend,
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
            backend=_backend,
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
            backend=_backend,
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
            backend=_backend,
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
            backend=_backend,
        )
        assert result.rmse >= 0.0


# ---------------------------------------------------------------------------
# marginal_value() tests
# ---------------------------------------------------------------------------


def _make_rows_with_candidate(n: int, seasons: list[int]) -> dict[int, list[dict[str, float | int]]]:
    """Build synthetic rows with a predictive feature_c and a noise column.

    target_slg = a*0.5 + b*0.3 + c*0.2
    target_avg = a*0.3 + c*0.1
    target_obp = a*0.4
    noise = deterministic pseudo-random, uncorrelated with targets
    """
    rows_by_season: dict[int, list[dict[str, float | int]]] = {}
    i = 0
    for season in seasons:
        season_rows: list[dict[str, float | int]] = []
        for _ in range(n):
            a = (i % 10) * 0.1
            b = ((i + 3) % 10) * 0.1
            c = ((i + 7) % 10) * 0.1
            noise = ((i * 7 + 13) % 17) / 17.0  # deterministic pseudo-random
            season_rows.append(
                {
                    "season": season,
                    "feature_a": a,
                    "feature_b": b,
                    "feature_c": c,
                    "noise": noise,
                    "target_slg": a * 0.5 + b * 0.3 + c * 0.2,
                    "target_avg": a * 0.3 + c * 0.1,
                    "target_obp": a * 0.4,
                }
            )
            i += 1
        rows_by_season[season] = season_rows
    return rows_by_season


class TestMarginalValuePredictiveCandidate:
    def test_feature_c_improves_slg(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023, 2024])
        result = marginal_value(
            candidate_column="feature_c",
            feature_columns=["feature_a", "feature_b"],
            targets=["slg"],
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
            backend=_backend,
        )
        assert result.candidate == "feature_c"
        # feature_c is predictive of slg, so at least one target should improve
        assert result.n_improved >= 1
        # The delta for slg should be negative (improvement)
        slg_delta = [d for d in result.deltas if d.target == "slg"][0]
        assert slg_delta.delta < 0


class TestMarginalValueNoiseCandidate:
    def test_noise_shows_near_zero_improvement(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023, 2024])
        result = marginal_value(
            candidate_column="noise",
            feature_columns=["feature_a", "feature_b"],
            targets=["slg"],
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
            backend=_backend,
        )
        assert result.candidate == "noise"
        # Noise should not substantially improve prediction (delta_pct > -5%)
        assert result.avg_delta_pct > -5.0


class TestMarginalValueMultiTarget:
    def test_evaluates_all_specified_targets(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023, 2024])
        result = marginal_value(
            candidate_column="feature_c",
            feature_columns=["feature_a", "feature_b"],
            targets=["slg", "avg", "obp"],
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
            backend=_backend,
        )
        assert result.n_total == 3
        target_names = {d.target for d in result.deltas}
        assert target_names == {"slg", "avg", "obp"}


class TestMarginalValueIdenticalData:
    def test_baseline_rmse_matches_quick_eval(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023, 2024])
        mv_result = marginal_value(
            candidate_column="feature_c",
            feature_columns=["feature_a", "feature_b"],
            targets=["slg"],
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
            backend=_backend,
        )
        qe_result = quick_eval(
            feature_columns=["feature_a", "feature_b"],
            target="slg",
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
            backend=_backend,
        )
        slg_delta = [d for d in mv_result.deltas if d.target == "slg"][0]
        assert abs(slg_delta.baseline_rmse - qe_result.rmse) < 1e-10


class TestMarginalValueMultiCandidateRanking:
    def test_predictive_candidate_ranks_first(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023, 2024])
        results = []
        for candidate in ["feature_c", "noise"]:
            results.append(
                marginal_value(
                    candidate_column=candidate,
                    feature_columns=["feature_a", "feature_b"],
                    targets=["slg"],
                    rows_by_season=rows,
                    train_seasons=[2022, 2023],
                    holdout_season=2024,
                    backend=_backend,
                )
            )
        ranked = sorted(results, key=lambda r: r.avg_delta_pct)
        # feature_c (predictive) should rank first (most negative avg_delta_pct)
        assert ranked[0].candidate == "feature_c"


class TestMarginalValueCustomParams:
    def test_params_forwarded_to_both_models(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023, 2024])
        result = marginal_value(
            candidate_column="feature_c",
            feature_columns=["feature_a", "feature_b"],
            targets=["slg"],
            rows_by_season=rows,
            train_seasons=[2022, 2023],
            holdout_season=2024,
            params={"max_iter": 50, "max_depth": 3},
            backend=_backend,
        )
        # Should complete successfully with custom params
        assert result.candidate == "feature_c"
        assert len(result.deltas) == 1


# ---------------------------------------------------------------------------
# compare_feature_sets() tests
# ---------------------------------------------------------------------------


class TestCompareFeatureSetsSingleHoldout:
    def test_set_b_superset_improves_slg(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023])
        result = compare_feature_sets(
            columns_a=["feature_a", "feature_b"],
            columns_b=["feature_a", "feature_b", "feature_c"],
            targets=["slg"],
            rows_by_season=rows,
            seasons=[2022, 2023],
            backend=_backend,
        )
        assert isinstance(result, FeatureSetComparisonResult)
        assert result.columns_a == ("feature_a", "feature_b")
        assert result.columns_b == ("feature_a", "feature_b", "feature_c")
        assert result.n_folds == 1
        assert result.n_total == 1
        # feature_c is predictive of slg, so B should win
        slg_delta = [d for d in result.deltas if d.target == "slg"][0]
        assert slg_delta.delta < 0
        assert result.n_improved >= 1

    def test_raises_with_fewer_than_2_seasons(self) -> None:
        rows = _make_rows_with_candidate(50, [2022])
        with pytest.raises(ValueError, match="at least 2 seasons"):
            compare_feature_sets(
                columns_a=["feature_a", "feature_b"],
                columns_b=["feature_a", "feature_b", "feature_c"],
                targets=["slg"],
                rows_by_season=rows,
                seasons=[2022],
                backend=_backend,
            )


class TestCompareFeatureSetsCVMode:
    def test_cv_mode_evaluates_all_targets(self) -> None:
        rows = _make_rows_with_candidate(50, [2021, 2022, 2023, 2024])
        result = compare_feature_sets(
            columns_a=["feature_a", "feature_b"],
            columns_b=["feature_a", "feature_b", "feature_c"],
            targets=["slg", "avg", "obp"],
            rows_by_season=rows,
            seasons=[2021, 2022, 2023, 2024],
            backend=_backend,
        )
        assert result.n_folds > 1
        assert result.n_total == 3
        target_names = {d.target for d in result.deltas}
        assert target_names == {"slg", "avg", "obp"}


class TestCompareFeatureSetsIdenticalSets:
    def test_identical_sets_near_zero_deltas(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023])
        result = compare_feature_sets(
            columns_a=["feature_a", "feature_b"],
            columns_b=["feature_a", "feature_b"],
            targets=["slg", "avg"],
            rows_by_season=rows,
            seasons=[2022, 2023],
            backend=_backend,
        )
        for d in result.deltas:
            assert abs(d.delta) < 1e-10
            assert abs(d.delta_pct) < 1e-6


class TestCompareFeatureSetsConsistentWithMarginalValue:
    def test_matches_marginal_value_for_single_feature_addition(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023])
        # marginal_value: train on [2022], holdout on 2023
        mv_result = marginal_value(
            candidate_column="feature_c",
            feature_columns=["feature_a", "feature_b"],
            targets=["slg"],
            rows_by_season=rows,
            train_seasons=[2022],
            holdout_season=2023,
            backend=_backend,
        )
        # compare_feature_sets with 2 seasons: single-holdout (train 2022, holdout 2023)
        cfs_result = compare_feature_sets(
            columns_a=["feature_a", "feature_b"],
            columns_b=["feature_a", "feature_b", "feature_c"],
            targets=["slg"],
            rows_by_season=rows,
            seasons=[2022, 2023],
            backend=_backend,
        )
        mv_slg = [d for d in mv_result.deltas if d.target == "slg"][0]
        cfs_slg = [d for d in cfs_result.deltas if d.target == "slg"][0]
        # Same train/holdout split → identical results
        assert abs(mv_slg.baseline_rmse - cfs_slg.baseline_rmse) < 1e-10
        assert abs(mv_slg.candidate_rmse - cfs_slg.candidate_rmse) < 1e-10
        assert abs(mv_slg.delta - cfs_slg.delta) < 1e-10


class TestCompareFeatureSetsCustomParams:
    def test_params_forwarded_to_both_models(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023])
        result = compare_feature_sets(
            columns_a=["feature_a", "feature_b"],
            columns_b=["feature_a", "feature_b", "feature_c"],
            targets=["slg"],
            rows_by_season=rows,
            seasons=[2022, 2023],
            params={"max_iter": 50, "max_depth": 3},
            backend=_backend,
        )
        assert result.n_total == 1
        assert len(result.deltas) == 1


class TestCompareFeatureSetsCVAveraging:
    def test_three_seasons_gives_two_folds(self) -> None:
        rows = _make_rows_with_candidate(50, [2022, 2023, 2024])
        result = compare_feature_sets(
            columns_a=["feature_a", "feature_b"],
            columns_b=["feature_a", "feature_b", "feature_c"],
            targets=["slg"],
            rows_by_season=rows,
            seasons=[2022, 2023, 2024],
            backend=_backend,
        )
        # 3 seasons → expanding CV: (train=[2022], test=2023), (train=[2022,2023], test=2024)
        assert result.n_folds == 2
