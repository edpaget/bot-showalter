import math
import random
from typing import Any

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from fantasy_baseball_manager.models.gbm_training import (
    CVFold,
    CorrelationGroup,
    FeatureImportance,
    GridSearchResult,
    GroupedImportanceResult,
    TargetVector,
    _evaluate_combination,
    _find_correlated_groups,
    build_cv_folds,
    compute_cv_permutation_importance,
    compute_grouped_permutation_importance,
    compute_permutation_importance,
    extract_features,
    extract_sample_weights,
    extract_targets,
    fit_models,
    grid_search_cv,
    identify_prune_candidates,
    score_predictions,
    sweep_cv,
    validate_pruning,
)
from fantasy_baseball_manager.models.protocols import (
    AblationResult,
    TargetComparison,
    ValidationResult,
)
from fantasy_baseball_manager.models.sample_weight_transforms import sqrt as sqrt_transform

pytestmark = pytest.mark.slow


class TestTargetComparison:
    def test_target_comparison_is_frozen(self) -> None:
        tc = TargetComparison(target="avg", full_rmse=0.03, pruned_rmse=0.031, delta_pct=3.3)
        assert tc.target == "avg"
        assert tc.full_rmse == 0.03
        assert tc.pruned_rmse == 0.031
        assert tc.delta_pct == 3.3
        with pytest.raises(AttributeError):
            tc.target = "other"  # type: ignore[misc]


class TestValidationResult:
    def test_validation_result_is_frozen(self) -> None:
        vr = ValidationResult(
            player_type="batter",
            comparisons=(),
            pruned_features=(),
            n_improved=0,
            n_degraded=0,
            max_degradation_pct=0.0,
            go=True,
        )
        assert vr.player_type == "batter"
        assert vr.go is True
        with pytest.raises(AttributeError):
            vr.go = False  # type: ignore[misc]


class TestAblationResultValidation:
    def test_ablation_result_validation_defaults_empty(self) -> None:
        result = AblationResult(model_name="x", feature_impacts={})
        assert result.validation_results == {}


class TestCorrelationGroup:
    def test_correlation_group_is_frozen(self) -> None:
        group = CorrelationGroup(name="group_0", members=("a", "b"))
        assert group.name == "group_0"
        assert group.members == ("a", "b")
        with pytest.raises(AttributeError):
            group.name = "other"  # type: ignore[misc]

    def test_members_is_tuple(self) -> None:
        group = CorrelationGroup(name="test", members=("x",))
        assert isinstance(group.members, tuple)


class TestGroupedImportanceResult:
    def test_grouped_importance_result_is_frozen(self) -> None:
        groups = (CorrelationGroup(name="a", members=("a",)),)
        fi = FeatureImportance(mean=0.1, se=0.01)
        result = GroupedImportanceResult(
            groups=groups,
            group_importance={"a": fi},
            feature_importance={"a": fi},
        )
        assert result.groups == groups
        assert result.group_importance == {"a": fi}
        assert result.feature_importance == {"a": fi}
        with pytest.raises(AttributeError):
            result.groups = ()  # type: ignore[misc]


class TestFindCorrelatedGroups:
    def test_perfectly_correlated_features_grouped(self) -> None:
        # col B = 2 * col A → same group
        X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
        groups = _find_correlated_groups(X, ["a", "b"])
        assert len(groups) == 1
        assert set(groups[0].members) == {"a", "b"}

    def test_uncorrelated_features_each_singleton(self) -> None:
        rng = random.Random(42)
        X = [[float(i), rng.random(), rng.random()] for i in range(50)]
        groups = _find_correlated_groups(X, ["a", "b", "c"])
        assert len(groups) == 3
        for g in groups:
            assert len(g.members) == 1

    def test_transitive_grouping(self) -> None:
        # A~B and B~C (via construction) but A~C not directly
        rng = random.Random(99)
        n = 100
        a = [float(i) for i in range(n)]
        b = [a[i] + rng.gauss(0, 0.5) for i in range(n)]
        c = [b[i] + rng.gauss(0, 0.5) for i in range(n)]
        X = [[a[i], b[i], c[i]] for i in range(n)]
        groups = _find_correlated_groups(X, ["a", "b", "c"], threshold=0.70)
        # All three should be in one group via transitivity
        multi_groups = [g for g in groups if len(g.members) > 1]
        assert len(multi_groups) == 1
        assert set(multi_groups[0].members) == {"a", "b", "c"}

    def test_threshold_boundary(self) -> None:
        # Pair at exactly threshold → not grouped (strict >)
        # Perfect correlation gives r=1.0, threshold=1.0 → not grouped
        X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
        groups = _find_correlated_groups(X, ["a", "b"], threshold=1.0)
        assert len(groups) == 2
        for g in groups:
            assert len(g.members) == 1

    def test_constant_column_is_singleton(self) -> None:
        # Constant column → NaN correlation → own group
        X = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]]
        groups = _find_correlated_groups(X, ["a", "const"])
        singletons = [g for g in groups if len(g.members) == 1]
        assert len(singletons) == 2

    def test_negative_correlation_groups(self) -> None:
        # A and -A → same group (abs correlation)
        X = [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]
        groups = _find_correlated_groups(X, ["a", "neg_a"])
        assert len(groups) == 1
        assert set(groups[0].members) == {"a", "neg_a"}

    def test_empty_feature_list(self) -> None:
        groups = _find_correlated_groups([], [], threshold=0.70)
        assert groups == []

    def test_single_feature_returns_singleton(self) -> None:
        X = [[1.0], [2.0], [3.0]]
        groups = _find_correlated_groups(X, ["only"])
        assert len(groups) == 1
        assert groups[0].members == ("only",)
        assert groups[0].name == "only"


class TestIdentifyPruneCandidates:
    def test_singleton_negative_ci_pruned(self) -> None:
        groups = (CorrelationGroup(name="noise", members=("noise",)),)
        result = GroupedImportanceResult(
            groups=groups,
            group_importance={"noise": FeatureImportance(mean=-0.01, se=0.002)},
            feature_importance={"noise": FeatureImportance(mean=-0.01, se=0.002)},
        )
        assert identify_prune_candidates(result) == ["noise"]

    def test_singleton_positive_ci_not_pruned(self) -> None:
        groups = (CorrelationGroup(name="signal", members=("signal",)),)
        result = GroupedImportanceResult(
            groups=groups,
            group_importance={"signal": FeatureImportance(mean=0.01, se=0.002)},
            feature_importance={"signal": FeatureImportance(mean=0.01, se=0.002)},
        )
        assert identify_prune_candidates(result) == []

    def test_multi_member_group_all_pruned(self) -> None:
        groups = (CorrelationGroup(name="group_0", members=("a", "b")),)
        result = GroupedImportanceResult(
            groups=groups,
            group_importance={"group_0": FeatureImportance(mean=-0.02, se=0.005)},
            feature_importance={
                "a": FeatureImportance(mean=-0.01, se=0.003),
                "b": FeatureImportance(mean=-0.01, se=0.003),
            },
        )
        assert identify_prune_candidates(result) == ["a", "b"]

    def test_multi_member_group_positive_ci_not_pruned(self) -> None:
        groups = (CorrelationGroup(name="group_0", members=("a", "b")),)
        result = GroupedImportanceResult(
            groups=groups,
            group_importance={"group_0": FeatureImportance(mean=0.05, se=0.01)},
            feature_importance={
                "a": FeatureImportance(mean=0.03, se=0.005),
                "b": FeatureImportance(mean=0.02, se=0.005),
            },
        )
        assert identify_prune_candidates(result) == []

    def test_boundary_zero_is_pruned(self) -> None:
        # mean + 2*SE = -0.004 + 2*0.002 = 0.0 → pruned
        groups = (CorrelationGroup(name="edge", members=("edge",)),)
        result = GroupedImportanceResult(
            groups=groups,
            group_importance={"edge": FeatureImportance(mean=-0.004, se=0.002)},
            feature_importance={"edge": FeatureImportance(mean=-0.004, se=0.002)},
        )
        assert identify_prune_candidates(result) == ["edge"]

    def test_empty_result_returns_empty(self) -> None:
        result = GroupedImportanceResult(
            groups=(),
            group_importance={},
            feature_importance={},
        )
        assert identify_prune_candidates(result) == []

    def test_mixed_groups(self) -> None:
        groups = (
            CorrelationGroup(name="good", members=("good",)),
            CorrelationGroup(name="bad", members=("bad",)),
        )
        result = GroupedImportanceResult(
            groups=groups,
            group_importance={
                "good": FeatureImportance(mean=0.05, se=0.01),
                "bad": FeatureImportance(mean=-0.02, se=0.005),
            },
            feature_importance={
                "good": FeatureImportance(mean=0.05, se=0.01),
                "bad": FeatureImportance(mean=-0.02, se=0.005),
            },
        )
        assert identify_prune_candidates(result) == ["bad"]

    def test_returns_sorted(self) -> None:
        groups = (
            CorrelationGroup(name="z_feat", members=("z_feat",)),
            CorrelationGroup(name="a_feat", members=("a_feat",)),
        )
        result = GroupedImportanceResult(
            groups=groups,
            group_importance={
                "z_feat": FeatureImportance(mean=-0.01, se=0.002),
                "a_feat": FeatureImportance(mean=-0.01, se=0.002),
            },
            feature_importance={
                "z_feat": FeatureImportance(mean=-0.01, se=0.002),
                "a_feat": FeatureImportance(mean=-0.01, se=0.002),
            },
        )
        assert identify_prune_candidates(result) == ["a_feat", "z_feat"]


class TestExtractTargets:
    def test_direct_stats(self) -> None:
        rows = [
            {"target_avg": 0.300, "target_obp": 0.380, "target_slg": 0.500, "target_woba": 0.370},
        ]
        result = extract_targets(rows, ["avg", "obp", "slg", "woba"])
        assert result["avg"].values == [0.300]
        assert result["avg"].indices == [0]
        assert result["obp"].values == [0.380]
        assert result["obp"].indices == [0]
        assert result["slg"].values == [0.500]
        assert result["slg"].indices == [0]
        assert result["woba"].values == [0.370]
        assert result["woba"].indices == [0]

    def test_iso_computed(self) -> None:
        rows = [
            {"target_slg": 0.500, "target_avg": 0.300},
        ]
        result = extract_targets(rows, ["iso"])
        assert len(result["iso"].values) == 1
        assert math.isclose(result["iso"].values[0], 0.200, abs_tol=1e-9)
        assert result["iso"].indices == [0]

    def test_babip_computed(self) -> None:
        # babip = (h - hr) / (ab - so - hr + sf)
        rows = [
            {"target_h": 150, "target_hr": 30, "target_ab": 500, "target_so": 100, "target_sf": 5},
        ]
        result = extract_targets(rows, ["babip"])
        expected = (150 - 30) / (500 - 100 - 30 + 5)
        assert len(result["babip"].values) == 1
        assert math.isclose(result["babip"].values[0], expected, abs_tol=1e-9)
        assert result["babip"].indices == [0]

    def test_skips_none(self) -> None:
        rows = [
            {"target_avg": 0.300},
            {"target_avg": None},
            {"target_avg": 0.250},
        ]
        result = extract_targets(rows, ["avg"])
        assert result["avg"].values == [0.300, 0.250]
        assert result["avg"].indices == [0, 2]

    def test_skips_missing_key(self) -> None:
        rows = [
            {"target_avg": 0.300},
            {},  # missing target_avg entirely
        ]
        result = extract_targets(rows, ["avg"])
        assert result["avg"].values == [0.300]
        assert result["avg"].indices == [0]

    def test_hr_per_9_computed(self) -> None:
        rows = [
            {"target_hr": 20, "target_ip": 180.0},
        ]
        result = extract_targets(rows, ["hr_per_9"])
        expected = 20 * 9 / 180.0
        assert len(result["hr_per_9"].values) == 1
        assert math.isclose(result["hr_per_9"].values[0], expected, abs_tol=1e-9)
        assert result["hr_per_9"].indices == [0]

    def test_hr_per_9_skips_zero_ip(self) -> None:
        rows = [
            {"target_hr": 5, "target_ip": 0},
        ]
        result = extract_targets(rows, ["hr_per_9"])
        assert result["hr_per_9"].values == []
        assert result["hr_per_9"].indices == []

    def test_pitcher_babip_computed(self) -> None:
        # pitcher babip = (h - hr) / (ip * 3 + h - so - hr)
        rows = [
            {"target_h": 150, "target_hr": 20, "target_ip": 180.0, "target_so": 160},
        ]
        result = extract_targets(rows, ["babip"])
        expected = (150 - 20) / (180.0 * 3 + 150 - 160 - 20)
        assert len(result["babip"].values) == 1
        assert math.isclose(result["babip"].values[0], expected, abs_tol=1e-9)
        assert result["babip"].indices == [0]

    def test_pitcher_babip_skips_zero_denom(self) -> None:
        # Construct a case where ip*3 + h - so - hr == 0
        rows = [
            {"target_h": 10, "target_hr": 5, "target_ip": 0, "target_so": 5},
        ]
        result = extract_targets(rows, ["babip"])
        assert result["babip"].values == []
        assert result["babip"].indices == []

    def test_different_targets_skip_different_rows(self) -> None:
        rows = [
            {"target_avg": 0.300, "target_slg": 0.500},
            {"target_avg": 0.280, "target_slg": None},
            {"target_avg": 0.250, "target_slg": 0.400},
        ]
        result = extract_targets(rows, ["avg", "iso"])
        assert result["avg"].values == [0.300, 0.280, 0.250]
        assert result["avg"].indices == [0, 1, 2]
        # iso needs both target_slg and target_avg; row 1 has target_slg=None
        assert len(result["iso"].values) == 2
        assert result["iso"].indices == [0, 2]


class TestExtractFeatures:
    def test_extracts_values(self) -> None:
        rows = [
            {"age": 28, "pa_1": 600, "hr_1": 30},
            {"age": 25, "pa_1": 550, "hr_1": 20},
        ]
        result = extract_features(rows, ["age", "pa_1", "hr_1"])
        assert result == [[28, 600, 30], [25, 550, 20]]

    def test_none_becomes_nan(self) -> None:
        rows = [{"age": 28, "pa_1": None}]
        result = extract_features(rows, ["age", "pa_1"])
        assert result[0][0] == 28
        assert math.isnan(result[0][1])

    def test_missing_key_becomes_nan(self) -> None:
        rows = [{"age": 28}]
        result = extract_features(rows, ["age", "pa_1"])
        assert result[0][0] == 28
        assert math.isnan(result[0][1])


class TestExtractSampleWeights:
    def test_returns_float_list(self) -> None:
        rows = [{"pa_1": 600}, {"pa_1": 400}, {"pa_1": 50}]
        result = extract_sample_weights(rows, "pa_1")
        assert result == [600.0, 400.0, 50.0]
        assert all(isinstance(v, float) for v in result)

    def test_missing_column_defaults_to_one(self) -> None:
        rows = [{"pa_1": 600}, {"other": 42}, {"pa_1": 50}]
        result = extract_sample_weights(rows, "pa_1")
        assert result == [600.0, 1.0, 50.0]

    def test_none_value_defaults_to_one(self) -> None:
        rows = [{"pa_1": 600}, {"pa_1": None}, {"pa_1": 50}]
        result = extract_sample_weights(rows, "pa_1")
        assert result == [600.0, 1.0, 50.0]


class TestFitModels:
    def test_returns_dict_of_models(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": TargetVector(indices=[0, 1, 2, 3], values=[0.250, 0.300, 0.275, 0.280])}
        models = fit_models(X, targets, {})
        assert "avg" in models
        assert hasattr(models["avg"], "predict")

    def test_respects_model_params(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": TargetVector(indices=[0, 1, 2, 3], values=[0.250, 0.300, 0.275, 0.280])}
        models = fit_models(X, targets, {"max_iter": 50})
        assert models["avg"].max_iter == 50

    def test_fit_with_partial_target_rows(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": TargetVector(indices=[0, 1, 3], values=[0.250, 0.300, 0.280])}
        models = fit_models(X, targets, {})
        assert "avg" in models
        assert hasattr(models["avg"], "predict")

    def test_fit_with_sample_weights(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": TargetVector(indices=[0, 1, 2, 3], values=[0.250, 0.300, 0.275, 0.280])}
        weights = [1.0, 2.0, 1.0, 3.0]
        models = fit_models(X, targets, {}, sample_weights=weights)
        assert "avg" in models
        assert hasattr(models["avg"], "predict")

    def test_fit_with_partial_targets_filters_weights(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": TargetVector(indices=[0, 1, 3], values=[0.250, 0.300, 0.280])}
        weights = [10.0, 20.0, 30.0, 40.0]
        models = fit_models(X, targets, {}, sample_weights=weights)
        assert "avg" in models
        assert hasattr(models["avg"], "predict")


class TestScorePredictions:
    def test_returns_rmse_keys(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {
            "avg": TargetVector(indices=[0, 1, 2, 3], values=[0.250, 0.300, 0.275, 0.280]),
            "obp": TargetVector(indices=[0, 1, 2, 3], values=[0.350, 0.400, 0.375, 0.380]),
        }
        models = fit_models(X, targets, {})
        metrics = score_predictions(models, X, targets)
        assert "rmse_avg" in metrics
        assert "rmse_obp" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_score_with_partial_target_rows(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        all_targets = {"avg": TargetVector(indices=[0, 1, 2, 3], values=[0.250, 0.300, 0.275, 0.280])}
        models = fit_models(X, all_targets, {})
        # Score with only 3 valid rows
        partial_targets = {"avg": TargetVector(indices=[0, 1, 3], values=[0.250, 0.300, 0.280])}
        metrics = score_predictions(models, X, partial_targets)
        assert "rmse_avg" in metrics
        assert isinstance(metrics["rmse_avg"], float)
        assert metrics["rmse_avg"] >= 0


class TestComputePermutationImportance:
    def test_returns_all_feature_columns(self) -> None:
        feature_cols = ["feat_a", "feat_b", "feat_c"]
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        targets = {"avg": TargetVector(indices=[0, 1, 2, 3], values=[0.250, 0.300, 0.275, 0.280])}
        models = fit_models(X, targets, {})
        result = compute_permutation_importance(models, X, targets, feature_cols)
        assert set(result.keys()) == set(feature_cols)

    def test_important_feature_has_positive_impact(self) -> None:
        # feat_a = target (signal); feat_b = noise
        # Use enough data + small min_samples_leaf so the tree actually learns
        gen = random.Random(99)
        all_X = [[float(i), gen.random()] for i in range(80)]
        all_y = [float(i) for i in range(80)]
        train_indices = list(range(0, 80, 2))
        holdout_indices = list(range(1, 80, 2))
        X_train = [all_X[i] for i in train_indices]
        y_train = {"y": TargetVector(indices=list(range(len(train_indices))), values=[all_y[i] for i in train_indices])}
        X_holdout = [all_X[i] for i in holdout_indices]
        y_holdout = {
            "y": TargetVector(indices=list(range(len(holdout_indices))), values=[all_y[i] for i in holdout_indices])
        }
        models = fit_models(X_train, y_train, {"min_samples_leaf": 5})
        result = compute_permutation_importance(models, X_holdout, y_holdout, ["feat_a", "feat_b"])
        assert result["feat_a"].mean > 0

    def test_irrelevant_feature_has_near_zero_impact(self) -> None:
        gen = random.Random(99)
        all_X = [[float(i), gen.random()] for i in range(80)]
        all_y = [float(i) for i in range(80)]
        train_indices = list(range(0, 80, 2))
        holdout_indices = list(range(1, 80, 2))
        X_train = [all_X[i] for i in train_indices]
        y_train = {"y": TargetVector(indices=list(range(len(train_indices))), values=[all_y[i] for i in train_indices])}
        X_holdout = [all_X[i] for i in holdout_indices]
        y_holdout = {
            "y": TargetVector(indices=list(range(len(holdout_indices))), values=[all_y[i] for i in holdout_indices])
        }
        models = fit_models(X_train, y_train, {"min_samples_leaf": 5})
        result = compute_permutation_importance(models, X_holdout, y_holdout, ["feat_a", "feat_b"])
        assert result["feat_b"].mean < result["feat_a"].mean

    def test_returns_feature_importance_instances(self) -> None:
        feature_cols = ["feat_a", "feat_b"]
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": TargetVector(indices=[0, 1, 2, 3], values=[0.250, 0.300, 0.275, 0.280])}
        models = fit_models(X, targets, {})
        result = compute_permutation_importance(models, X, targets, feature_cols)
        for fi in result.values():
            assert isinstance(fi, FeatureImportance)

    def test_standard_error_is_non_negative(self) -> None:
        feature_cols = ["feat_a", "feat_b"]
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": TargetVector(indices=[0, 1, 2, 3], values=[0.250, 0.300, 0.275, 0.280])}
        models = fit_models(X, targets, {})
        result = compute_permutation_importance(models, X, targets, feature_cols)
        for fi in result.values():
            assert fi.se >= 0

    def test_single_repeat_has_zero_se(self) -> None:
        feature_cols = ["feat_a", "feat_b"]
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": TargetVector(indices=[0, 1, 2, 3], values=[0.250, 0.300, 0.275, 0.280])}
        models = fit_models(X, targets, {})
        result = compute_permutation_importance(models, X, targets, feature_cols, n_repeats=1)
        for fi in result.values():
            assert fi.se == 0.0


class TestComputeGroupedPermutationImportance:
    def test_returns_all_groups_and_features(self) -> None:
        gen = random.Random(42)
        # 3 features: a (signal), b = 2*a + noise (correlated), c (noise)
        n = 60
        a_vals = [float(i) for i in range(n)]
        b_vals = [2 * a_vals[i] + gen.gauss(0, 0.1) for i in range(n)]
        c_vals = [gen.random() for _ in range(n)]
        X = [[a_vals[i], b_vals[i], c_vals[i]] for i in range(n)]
        targets = {"y": TargetVector(indices=list(range(n)), values=a_vals)}
        models = fit_models(X, targets, {"min_samples_leaf": 5})
        result = compute_grouped_permutation_importance(models, X, targets, ["a", "b", "c"], n_repeats=5)
        assert isinstance(result, GroupedImportanceResult)
        # All feature columns present in feature_importance
        assert set(result.feature_importance.keys()) == {"a", "b", "c"}
        # All group names present in group_importance
        assert set(result.group_importance.keys()) == {g.name for g in result.groups}

    def test_correlated_group_importance_higher_than_individual(self) -> None:
        gen = random.Random(42)
        n = 80
        a_vals = [float(i) for i in range(n)]
        b_vals = [a_vals[i] + gen.gauss(0, 0.3) for i in range(n)]
        c_vals = [gen.random() for _ in range(n)]
        X = [[a_vals[i], b_vals[i], c_vals[i]] for i in range(n)]
        targets = {"y": TargetVector(indices=list(range(n)), values=a_vals)}
        models = fit_models(X, targets, {"min_samples_leaf": 5})
        result = compute_grouped_permutation_importance(models, X, targets, ["a", "b", "c"], n_repeats=10)
        # Find the multi-member group
        multi_groups = [g for g in result.groups if len(g.members) > 1]
        assert len(multi_groups) >= 1
        group = multi_groups[0]
        group_imp = result.group_importance[group.name].mean
        # Group importance should exceed individual importance of either member
        for member in group.members:
            assert group_imp >= result.feature_importance[member].mean

    def test_singleton_group_matches_individual(self) -> None:
        gen = random.Random(42)
        n = 40
        X = [[float(i), gen.random()] for i in range(n)]
        targets = {"y": TargetVector(indices=list(range(n)), values=[float(i) for i in range(n)])}
        models = fit_models(X, targets, {"min_samples_leaf": 5})
        result = compute_grouped_permutation_importance(
            models, X, targets, ["a", "b"], n_repeats=5, correlation_threshold=0.99
        )
        # With high threshold, all should be singletons
        for g in result.groups:
            assert len(g.members) == 1
            feat_name = g.members[0]
            assert math.isclose(
                result.group_importance[g.name].mean,
                result.feature_importance[feat_name].mean,
                rel_tol=1e-9,
            )

    def test_all_values_are_feature_importance(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"y": TargetVector(indices=[0, 1, 2, 3], values=[1.0, 2.0, 3.0, 4.0])}
        models = fit_models(X, targets, {})
        result = compute_grouped_permutation_importance(models, X, targets, ["a", "b"], n_repeats=3)
        for fi in result.group_importance.values():
            assert isinstance(fi, FeatureImportance)
        for fi in result.feature_importance.values():
            assert isinstance(fi, FeatureImportance)

    def test_standard_errors_non_negative(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"y": TargetVector(indices=[0, 1, 2, 3], values=[1.0, 2.0, 3.0, 4.0])}
        models = fit_models(X, targets, {})
        result = compute_grouped_permutation_importance(models, X, targets, ["a", "b"], n_repeats=5)
        for fi in result.group_importance.values():
            assert fi.se >= 0
        for fi in result.feature_importance.values():
            assert fi.se >= 0

    def test_custom_correlation_threshold(self) -> None:
        # threshold=1.0 (strict >) → all singletons even for perfectly correlated
        X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
        targets = {"y": TargetVector(indices=[0, 1, 2, 3], values=[1.0, 2.0, 3.0, 4.0])}
        models = fit_models(X, targets, {})
        result_high = compute_grouped_permutation_importance(
            models, X, targets, ["a", "b"], n_repeats=3, correlation_threshold=1.0
        )
        assert all(len(g.members) == 1 for g in result_high.groups)

        # Very low threshold → one big group
        result_low = compute_grouped_permutation_importance(
            models, X, targets, ["a", "b"], n_repeats=3, correlation_threshold=0.01
        )
        multi = [g for g in result_low.groups if len(g.members) > 1]
        assert len(multi) == 1

    def test_single_repeat_zero_se(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"y": TargetVector(indices=[0, 1, 2, 3], values=[1.0, 2.0, 3.0, 4.0])}
        models = fit_models(X, targets, {})
        result = compute_grouped_permutation_importance(models, X, targets, ["a", "b"], n_repeats=1)
        for fi in result.group_importance.values():
            assert fi.se == 0.0
        for fi in result.feature_importance.values():
            assert fi.se == 0.0

    def test_explicit_groups_override_detection(self) -> None:
        gen = random.Random(42)
        n = 60
        a_vals = [float(i) for i in range(n)]
        b_vals = [2 * a_vals[i] + gen.gauss(0, 0.1) for i in range(n)]
        c_vals = [gen.random() for _ in range(n)]
        X = [[a_vals[i], b_vals[i], c_vals[i]] for i in range(n)]
        targets = {"y": TargetVector(indices=list(range(n)), values=a_vals)}
        models = fit_models(X, targets, {"min_samples_leaf": 5})
        # Provide explicit groups: treat all 3 as singletons (override auto-detection)
        explicit_groups = [
            CorrelationGroup(name="a", members=("a",)),
            CorrelationGroup(name="b", members=("b",)),
            CorrelationGroup(name="c", members=("c",)),
        ]
        result = compute_grouped_permutation_importance(
            models, X, targets, ["a", "b", "c"], n_repeats=5, groups=explicit_groups
        )
        # All 3 should be singletons (our explicit groups), not auto-detected correlation groups
        assert len(result.groups) == 3
        for g in result.groups:
            assert len(g.members) == 1


class TestEndToEndWithMissingTargets:
    def test_extract_fit_score_with_missing_targets(self) -> None:
        rows = [
            {"feat_a": 1.0, "feat_b": 2.0, "target_avg": 0.300, "target_slg": 0.500},
            {"feat_a": 3.0, "feat_b": 4.0, "target_avg": 0.280, "target_slg": None},
            {"feat_a": 5.0, "feat_b": 6.0, "target_avg": 0.250, "target_slg": 0.400},
            {"feat_a": 7.0, "feat_b": 8.0, "target_avg": 0.270, "target_slg": 0.450},
        ]
        X = extract_features(rows, ["feat_a", "feat_b"])
        targets = extract_targets(rows, ["avg", "iso"])
        # avg valid for all 4 rows, iso missing for row 1
        assert len(targets["avg"].values) == 4
        assert len(targets["iso"].values) == 3
        models = fit_models(X, targets, {})
        metrics = score_predictions(models, X, targets)
        assert "rmse_avg" in metrics
        assert "rmse_iso" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestBuildCVFolds:
    def _make_rows(self, season: int, n: int = 5) -> list[dict[str, Any]]:
        return [
            {"season": season, "feat_a": float(i), "feat_b": float(i * 2), "target_y": float(i * 0.1), "pa_1": 500 + i}
            for i in range(n)
        ]

    def test_groups_by_season(self) -> None:
        rows = self._make_rows(2021, 3) + self._make_rows(2022, 4) + self._make_rows(2023, 5)
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022), ([2021, 2022], 2023)]
        folds = build_cv_folds(rows, ["feat_a", "feat_b"], ["y"], cv_splits)
        assert len(folds) == 2
        # Fold 0: train=2021 (3 rows), test=2022 (4 rows)
        assert len(folds[0].X_train) == 3
        assert len(folds[0].X_test) == 4
        # Fold 1: train=2021+2022 (7 rows), test=2023 (5 rows)
        assert len(folds[1].X_train) == 7
        assert len(folds[1].X_test) == 5

    def test_with_sample_weights(self) -> None:
        rows = self._make_rows(2021, 3) + self._make_rows(2022, 3)
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(rows, ["feat_a", "feat_b"], ["y"], cv_splits, sample_weight_column="pa_1")
        assert folds[0].sample_weights is not None
        assert len(folds[0].sample_weights) == 3

    def test_without_sample_weights(self) -> None:
        rows = self._make_rows(2021, 3) + self._make_rows(2022, 3)
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(rows, ["feat_a", "feat_b"], ["y"], cv_splits)
        assert folds[0].sample_weights is None

    def test_empty_season(self) -> None:
        rows = self._make_rows(2021, 3)
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]  # 2022 has no rows
        folds = build_cv_folds(rows, ["feat_a", "feat_b"], ["y"], cv_splits)
        assert len(folds) == 1
        assert len(folds[0].X_train) == 3
        assert len(folds[0].X_test) == 0

    def test_with_transform(self) -> None:
        rows = self._make_rows(2021, 3) + self._make_rows(2022, 3)
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            sample_weight_column="pa_1",
            sample_weight_transform=sqrt_transform,
        )
        assert folds[0].sample_weights is not None
        raw_weights = extract_sample_weights([r for r in rows if r["season"] == 2021], "pa_1")
        for actual, raw_w in zip(folds[0].sample_weights, raw_weights):
            assert math.isclose(actual, math.sqrt(raw_w), abs_tol=1e-9)

    def test_top_n_filters_test_rows_by_weight(self) -> None:
        rows = self._make_rows(2021, 5) + self._make_rows(2022, 5)
        # Give rows different pa_1 values so we can verify filtering
        for i, row in enumerate(r for r in rows if r["season"] == 2022):
            row["pa_1"] = 100 + i * 100  # 100, 200, 300, 400, 500
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(rows, ["feat_a", "feat_b"], ["y"], cv_splits, sample_weight_column="pa_1", test_top_n=3)
        assert len(folds[0].X_test) == 3

    def test_top_n_none_includes_all(self) -> None:
        rows = self._make_rows(2021, 3) + self._make_rows(2022, 5)
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(
            rows, ["feat_a", "feat_b"], ["y"], cv_splits, sample_weight_column="pa_1", test_top_n=None
        )
        assert len(folds[0].X_test) == 5

    def test_top_n_larger_than_fold_includes_all(self) -> None:
        rows = self._make_rows(2021, 3) + self._make_rows(2022, 5)
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(
            rows, ["feat_a", "feat_b"], ["y"], cv_splits, sample_weight_column="pa_1", test_top_n=100
        )
        assert len(folds[0].X_test) == 5

    def test_top_n_requires_sample_weight_or_rank_column(self) -> None:
        rows = self._make_rows(2021, 3) + self._make_rows(2022, 3)
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        with pytest.raises(ValueError, match="test_top_n requires sample_weight_column or test_rank_column"):
            build_cv_folds(rows, ["feat_a", "feat_b"], ["y"], cv_splits, test_top_n=3)

    def test_top_n_does_not_affect_train_rows(self) -> None:
        rows = self._make_rows(2021, 10) + self._make_rows(2022, 10)
        for i, row in enumerate(r for r in rows if r["season"] == 2022):
            row["pa_1"] = 100 + i * 50
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(rows, ["feat_a", "feat_b"], ["y"], cv_splits, sample_weight_column="pa_1", test_top_n=3)
        assert len(folds[0].X_train) == 10
        assert len(folds[0].X_test) == 3

    def test_rank_column_filters_test_rows(self) -> None:
        """test_rank_column ranks test rows by the specified column, not sample_weight_column."""
        rows: list[dict[str, Any]] = []
        for i in range(5):
            rows.append(
                {
                    "season": 2021,
                    "feat_a": float(i),
                    "feat_b": float(i * 2),
                    "target_y": float(i * 0.1),
                    "pa_1": 500 + i,
                    "war": float(i),
                }
            )
        for i in range(5):
            # war and pa_1 diverge: pa_1 is always 500 but war varies
            rows.append(
                {
                    "season": 2022,
                    "feat_a": float(i),
                    "feat_b": float(i * 2),
                    "target_y": float(i * 0.1),
                    "pa_1": 500,
                    "war": float(i * 2),  # 0, 2, 4, 6, 8
                }
            )
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            sample_weight_column="pa_1",
            test_top_n=3,
            test_rank_column="war",
        )
        assert len(folds[0].X_test) == 3

    def test_rank_column_defaults_to_sample_weight_column(self) -> None:
        """Without test_rank_column, test_top_n filters by sample_weight_column."""
        rows = self._make_rows(2021, 5) + self._make_rows(2022, 5)
        for i, row in enumerate(r for r in rows if r["season"] == 2022):
            row["pa_1"] = 100 + i * 100
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(rows, ["feat_a", "feat_b"], ["y"], cv_splits, sample_weight_column="pa_1", test_top_n=3)
        assert len(folds[0].X_test) == 3

    def test_rank_column_without_top_n_is_noop(self) -> None:
        """test_rank_column without test_top_n includes all test rows."""
        rows: list[dict[str, Any]] = []
        for i in range(5):
            rows.append(
                {
                    "season": 2021,
                    "feat_a": float(i),
                    "feat_b": float(i * 2),
                    "target_y": float(i * 0.1),
                    "pa_1": 500 + i,
                    "war": float(i),
                }
            )
        for i in range(5):
            rows.append(
                {
                    "season": 2022,
                    "feat_a": float(i),
                    "feat_b": float(i * 2),
                    "target_y": float(i * 0.1),
                    "pa_1": 500,
                    "war": float(i),
                }
            )
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            test_rank_column="war",
        )
        assert len(folds[0].X_test) == 5

    def test_rank_column_alone_allows_top_n(self) -> None:
        """test_rank_column without sample_weight_column still allows test_top_n."""
        rows: list[dict[str, Any]] = []
        for i in range(5):
            rows.append(
                {
                    "season": 2021,
                    "feat_a": float(i),
                    "feat_b": float(i * 2),
                    "target_y": float(i * 0.1),
                    "war": float(i),
                }
            )
        for i in range(5):
            rows.append(
                {
                    "season": 2022,
                    "feat_a": float(i),
                    "feat_b": float(i * 2),
                    "target_y": float(i * 0.1),
                    "war": float(i * 2),
                }
            )
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds = build_cv_folds(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            test_top_n=3,
            test_rank_column="war",
        )
        assert len(folds[0].X_test) == 3

    def test_transform_none_no_effect(self) -> None:
        rows = self._make_rows(2021, 3) + self._make_rows(2022, 3)
        cv_splits: list[tuple[list[int], int]] = [([2021], 2022)]
        folds_no_transform = build_cv_folds(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            sample_weight_column="pa_1",
        )
        folds_none = build_cv_folds(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            sample_weight_column="pa_1",
            sample_weight_transform=None,
        )
        assert folds_no_transform[0].sample_weights == folds_none[0].sample_weights


def _make_cv_folds() -> list[CVFold]:
    """Build 2 simple CV folds with linear signal in feat_a."""
    gen = random.Random(42)
    # 3 seasons of data: train on expanding window, test on next
    season_data: dict[int, tuple[list[list[float]], list[float]]] = {}
    for s in [2020, 2021, 2022]:
        X = [[float(i + s * 10), gen.random()] for i in range(10)]
        y = [float(i + s * 10) for i in range(10)]
        season_data[s] = (X, y)

    # Fold 0: train=2020, test=2021
    fold0_X_train, fold0_y_train_vals = season_data[2020]
    fold0_X_test, fold0_y_test_vals = season_data[2021]
    fold0 = CVFold(
        X_train=fold0_X_train,
        y_train={"y": TargetVector(list(range(len(fold0_y_train_vals))), fold0_y_train_vals)},
        X_test=fold0_X_test,
        y_test={"y": TargetVector(list(range(len(fold0_y_test_vals))), fold0_y_test_vals)},
    )

    # Fold 1: train=2020+2021, test=2022
    fold1_X_train = season_data[2020][0] + season_data[2021][0]
    fold1_y_train_vals = season_data[2020][1] + season_data[2021][1]
    fold1_X_test, fold1_y_test_vals = season_data[2022]
    fold1 = CVFold(
        X_train=fold1_X_train,
        y_train={"y": TargetVector(list(range(len(fold1_y_train_vals))), fold1_y_train_vals)},
        X_test=fold1_X_test,
        y_test={"y": TargetVector(list(range(len(fold1_y_test_vals))), fold1_y_test_vals)},
    )

    return [fold0, fold1]


class TestGridSearchCV:
    def test_single_param_returns_better_option(self) -> None:
        folds = _make_cv_folds()
        # More iterations should generally do at least as well
        result = grid_search_cv(folds, {"max_iter": [50, 100]})
        assert result.best_params["max_iter"] in [50, 100]

    def test_best_params_contains_all_grid_keys(self) -> None:
        folds = _make_cv_folds()
        result = grid_search_cv(folds, {"max_iter": [100], "max_depth": [3, 5]})
        assert "max_iter" in result.best_params
        assert "max_depth" in result.best_params

    def test_per_target_rmse_has_one_entry_per_target(self) -> None:
        folds = _make_cv_folds()
        result = grid_search_cv(folds, {"max_iter": [100]})
        assert "y" in result.per_target_rmse
        assert len(result.per_target_rmse) == 1

    def test_all_results_has_one_entry_per_combination(self) -> None:
        folds = _make_cv_folds()
        result = grid_search_cv(folds, {"max_iter": [50, 100], "max_depth": [3, 5]})
        # 2 * 2 = 4 combinations
        assert len(result.all_results) == 4

    def test_trivial_single_combination(self) -> None:
        folds = _make_cv_folds()
        result = grid_search_cv(folds, {"max_iter": [100]})
        assert result.best_params == {"max_iter": 100}
        assert len(result.all_results) == 1

    def test_result_is_grid_search_result(self) -> None:
        folds = _make_cv_folds()
        result = grid_search_cv(folds, {"max_iter": [100]})
        assert isinstance(result, GridSearchResult)
        assert isinstance(result.best_mean_rmse, float)
        assert result.best_mean_rmse >= 0

    def test_evaluate_combination_returns_valid_result(self) -> None:
        folds = _make_cv_folds()
        params = {"max_iter": 100}
        entry = _evaluate_combination(folds, params)
        assert entry["params"] == params
        assert isinstance(entry["mean_rmse"], float)
        assert entry["mean_rmse"] >= 0
        assert "y" in entry["per_target_rmse"]

    def test_evaluate_combination_with_weights(self) -> None:
        folds = _make_cv_folds()
        # Add sample_weights to each fold
        weighted_folds = [
            CVFold(
                X_train=f.X_train,
                y_train=f.y_train,
                X_test=f.X_test,
                y_test=f.y_test,
                sample_weights=[1.0] * len(f.X_train),
            )
            for f in folds
        ]
        entry = _evaluate_combination(weighted_folds, {"max_iter": 100})
        assert isinstance(entry["mean_rmse"], float)
        assert entry["mean_rmse"] >= 0

    def test_grid_search_with_sample_weights(self) -> None:
        folds = _make_cv_folds()
        weighted_folds = [
            CVFold(
                X_train=f.X_train,
                y_train=f.y_train,
                X_test=f.X_test,
                y_test=f.y_test,
                sample_weights=[1.0] * len(f.X_train),
            )
            for f in folds
        ]
        result = grid_search_cv(weighted_folds, {"max_iter": [100]})
        assert isinstance(result, GridSearchResult)
        assert result.best_mean_rmse >= 0

    def test_parallel_matches_sequential(self) -> None:
        folds = _make_cv_folds()
        param_grid = {"max_iter": [50, 100], "max_depth": [3, 5]}
        sequential = grid_search_cv(folds, param_grid, max_workers=1)
        parallel = grid_search_cv(folds, param_grid, max_workers=2)
        assert sequential.best_params == parallel.best_params
        assert len(sequential.all_results) == len(parallel.all_results)
        for seq_entry, par_entry in zip(sequential.all_results, parallel.all_results):
            assert seq_entry["params"] == par_entry["params"]
            assert math.isclose(seq_entry["mean_rmse"], par_entry["mean_rmse"], rel_tol=1e-9)

    def test_max_workers_defaults_to_none(self) -> None:
        folds = _make_cv_folds()
        result = grid_search_cv(folds, {"max_iter": [100]})
        assert isinstance(result, GridSearchResult)
        assert result.best_mean_rmse >= 0


def _make_validate_data() -> tuple[
    dict[str, HistGradientBoostingRegressor],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[str],
]:
    """Build synthetic data with signal feature (feat_a) and noise feature (feat_noise)."""
    gen = random.Random(42)
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for i in range(60):
        val = float(i)
        row: dict[str, Any] = {
            "feat_a": val,
            "feat_noise": gen.random(),
            "target_y": val * 0.01,
        }
        if i < 40:
            train_rows.append(row)
        else:
            holdout_rows.append(row)
    feature_cols = ["feat_a", "feat_noise"]
    targets = ["y"]
    X_train = extract_features(train_rows, feature_cols)
    y_train = extract_targets(train_rows, targets)
    full_models = fit_models(X_train, y_train, {"min_samples_leaf": 5})
    return full_models, train_rows, holdout_rows, feature_cols, targets


class TestValidatePruning:
    def test_returns_validation_result(self) -> None:
        full_models, train_rows, holdout_rows, feature_cols, targets = _make_validate_data()
        result = validate_pruning(
            full_models,
            train_rows,
            holdout_rows,
            feature_cols,
            prune_set=["feat_noise"],
            targets=targets,
            model_params={"min_samples_leaf": 5},
            player_type="batter",
        )
        assert isinstance(result, ValidationResult)

    def test_comparisons_cover_all_targets(self) -> None:
        full_models, train_rows, holdout_rows, feature_cols, targets = _make_validate_data()
        result = validate_pruning(
            full_models,
            train_rows,
            holdout_rows,
            feature_cols,
            prune_set=["feat_noise"],
            targets=targets,
            model_params={"min_samples_leaf": 5},
            player_type="batter",
        )
        comparison_targets = [c.target for c in result.comparisons]
        assert comparison_targets == targets

    def test_delta_pct_is_relative_to_full(self) -> None:
        full_models, train_rows, holdout_rows, feature_cols, targets = _make_validate_data()
        result = validate_pruning(
            full_models,
            train_rows,
            holdout_rows,
            feature_cols,
            prune_set=["feat_noise"],
            targets=targets,
            model_params={"min_samples_leaf": 5},
            player_type="batter",
        )
        for comp in result.comparisons:
            expected_pct = (comp.pruned_rmse - comp.full_rmse) / comp.full_rmse * 100
            assert math.isclose(comp.delta_pct, expected_pct, rel_tol=1e-9)

    def test_go_when_pruning_noise_feature(self) -> None:
        full_models, train_rows, holdout_rows, feature_cols, targets = _make_validate_data()
        result = validate_pruning(
            full_models,
            train_rows,
            holdout_rows,
            feature_cols,
            prune_set=["feat_noise"],
            targets=targets,
            model_params={"min_samples_leaf": 5},
            player_type="batter",
        )
        # Pruning noise should not degrade, so go should be True
        assert result.go is True

    def test_nogo_when_pruning_important_feature(self) -> None:
        full_models, train_rows, holdout_rows, feature_cols, targets = _make_validate_data()
        result = validate_pruning(
            full_models,
            train_rows,
            holdout_rows,
            feature_cols,
            prune_set=["feat_a"],
            targets=targets,
            model_params={"min_samples_leaf": 5},
            player_type="batter",
        )
        # Pruning the signal feature should degrade performance
        assert result.go is False

    def test_empty_prune_set(self) -> None:
        full_models, train_rows, holdout_rows, feature_cols, targets = _make_validate_data()
        result = validate_pruning(
            full_models,
            train_rows,
            holdout_rows,
            feature_cols,
            prune_set=[],
            targets=targets,
            model_params={"min_samples_leaf": 5},
            player_type="batter",
        )
        assert result.go is True
        assert result.comparisons == ()
        assert result.pruned_features == ()

    def test_records_pruned_features(self) -> None:
        full_models, train_rows, holdout_rows, feature_cols, targets = _make_validate_data()
        result = validate_pruning(
            full_models,
            train_rows,
            holdout_rows,
            feature_cols,
            prune_set=["feat_noise"],
            targets=targets,
            model_params={"min_samples_leaf": 5},
            player_type="batter",
        )
        assert result.pruned_features == ("feat_noise",)

    def test_custom_max_degradation_pct(self) -> None:
        full_models, train_rows, holdout_rows, feature_cols, targets = _make_validate_data()
        # Prune the important feature with a very tight threshold
        result = validate_pruning(
            full_models,
            train_rows,
            holdout_rows,
            feature_cols,
            prune_set=["feat_a"],
            targets=targets,
            model_params={"min_samples_leaf": 5},
            player_type="batter",
            max_degradation_pct=0.0,
        )
        # Should be NO-GO since removing the signal feature degrades
        assert result.go is False


def _make_cv_importance_data() -> dict[int, list[dict[str, Any]]]:
    """Build 3 seasons of test data: feat_a = linear signal, feat_b = noise."""
    gen = random.Random(42)
    rows_by_season: dict[int, list[dict[str, Any]]] = {}
    for season in [2020, 2021, 2022]:
        rows: list[dict[str, Any]] = []
        for i in range(20):
            val = float(i + season * 10)
            rows.append(
                {
                    "feat_a": val,
                    "feat_b": gen.random(),
                    "target_y": val * 0.01,
                }
            )
        rows_by_season[season] = rows
    return rows_by_season


class TestComputeCVPermutationImportance:
    def test_returns_grouped_importance_result(self) -> None:
        data = _make_cv_importance_data()
        result = compute_cv_permutation_importance(
            data, ["feat_a", "feat_b"], ["y"], {"min_samples_leaf": 5}, n_repeats=5
        )
        assert isinstance(result, GroupedImportanceResult)

    def test_returns_all_feature_columns(self) -> None:
        data = _make_cv_importance_data()
        result = compute_cv_permutation_importance(
            data, ["feat_a", "feat_b"], ["y"], {"min_samples_leaf": 5}, n_repeats=5
        )
        assert set(result.feature_importance.keys()) == {"feat_a", "feat_b"}

    def test_signal_feature_has_positive_importance(self) -> None:
        data = _make_cv_importance_data()
        result = compute_cv_permutation_importance(
            data, ["feat_a", "feat_b"], ["y"], {"min_samples_leaf": 5}, n_repeats=5
        )
        assert result.feature_importance["feat_a"].mean > 0

    def test_noise_feature_less_important_than_signal(self) -> None:
        data = _make_cv_importance_data()
        result = compute_cv_permutation_importance(
            data, ["feat_a", "feat_b"], ["y"], {"min_samples_leaf": 5}, n_repeats=5
        )
        assert result.feature_importance["feat_b"].mean < result.feature_importance["feat_a"].mean

    def test_se_non_negative(self) -> None:
        data = _make_cv_importance_data()
        result = compute_cv_permutation_importance(
            data, ["feat_a", "feat_b"], ["y"], {"min_samples_leaf": 5}, n_repeats=5
        )
        for fi in result.feature_importance.values():
            assert fi.se >= 0
        for gi in result.group_importance.values():
            assert gi.se >= 0

    def test_groups_present_in_result(self) -> None:
        data = _make_cv_importance_data()
        result = compute_cv_permutation_importance(
            data, ["feat_a", "feat_b"], ["y"], {"min_samples_leaf": 5}, n_repeats=5
        )
        assert len(result.groups) > 0

    def test_two_seasons_single_fold(self) -> None:
        gen = random.Random(99)
        data: dict[int, list[dict[str, Any]]] = {}
        for season in [2021, 2022]:
            rows: list[dict[str, Any]] = []
            for i in range(20):
                val = float(i + season * 10)
                rows.append(
                    {
                        "feat_a": val,
                        "feat_b": gen.random(),
                        "target_y": val * 0.01,
                    }
                )
            data[season] = rows
        result = compute_cv_permutation_importance(
            data, ["feat_a", "feat_b"], ["y"], {"min_samples_leaf": 5}, n_repeats=5
        )
        assert isinstance(result, GroupedImportanceResult)

    def test_single_season_raises(self) -> None:
        gen = random.Random(42)
        data = {2022: [{"feat_a": float(i), "feat_b": gen.random(), "target_y": float(i)} for i in range(20)]}
        with pytest.raises(ValueError, match="at least 2 seasons"):
            compute_cv_permutation_importance(data, ["feat_a", "feat_b"], ["y"], {})

    def test_empty_seasons_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 seasons"):
            compute_cv_permutation_importance({}, ["feat_a", "feat_b"], ["y"], {})

    def test_reproducible_with_same_seed(self) -> None:
        data = _make_cv_importance_data()
        result1 = compute_cv_permutation_importance(
            data, ["feat_a", "feat_b"], ["y"], {"min_samples_leaf": 5}, n_repeats=5, rng_seed=42
        )
        result2 = compute_cv_permutation_importance(
            data, ["feat_a", "feat_b"], ["y"], {"min_samples_leaf": 5}, n_repeats=5, rng_seed=42
        )
        for col in ["feat_a", "feat_b"]:
            assert result1.feature_importance[col].mean == result2.feature_importance[col].mean
            assert result1.feature_importance[col].se == result2.feature_importance[col].se


def _make_sweep_rows() -> list[dict[str, Any]]:
    """Build 3 seasons of rows with a weight column for sweep tests."""
    gen = random.Random(42)
    rows: list[dict[str, Any]] = []
    for season in [2020, 2021, 2022]:
        for i in range(10):
            val = float(i + season * 10)
            rows.append(
                {
                    "season": season,
                    "feat_a": val,
                    "feat_b": gen.random(),
                    "target_y": val * 0.01,
                    "pa_1": 100 + i * 30,
                }
            )
    return rows


class TestSweepCV:
    def test_returns_best_transform(self) -> None:
        rows = _make_sweep_rows()
        cv_splits: list[tuple[list[int], int]] = [([2020], 2021), ([2020, 2021], 2022)]
        result = sweep_cv(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            {},
            {"sample_weight_transform": ["raw", "sqrt"]},
            sample_weight_column="pa_1",
        )
        assert isinstance(result, GridSearchResult)
        assert "sample_weight_transform" in result.best_params
        assert result.best_params["sample_weight_transform"] in ["raw", "sqrt"]

    def test_all_results_populated(self) -> None:
        rows = _make_sweep_rows()
        cv_splits: list[tuple[list[int], int]] = [([2020], 2021), ([2020, 2021], 2022)]
        result = sweep_cv(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            {},
            {"sample_weight_transform": ["raw", "sqrt", "log1p"]},
            sample_weight_column="pa_1",
        )
        assert len(result.all_results) == 3

    def test_single_combo(self) -> None:
        rows = _make_sweep_rows()
        cv_splits: list[tuple[list[int], int]] = [([2020], 2021), ([2020, 2021], 2022)]
        result = sweep_cv(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            {},
            {"sample_weight_transform": ["sqrt"]},
            sample_weight_column="pa_1",
        )
        assert result.best_params == {"sample_weight_transform": "sqrt"}
        assert len(result.all_results) == 1

    def test_no_weight_column_ignores_transform(self) -> None:
        rows = _make_sweep_rows()
        cv_splits: list[tuple[list[int], int]] = [([2020], 2021), ([2020, 2021], 2022)]
        result = sweep_cv(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            {},
            {"sample_weight_transform": ["raw", "sqrt"]},
            sample_weight_column=None,
        )
        # Without weights, all transforms yield the same RMSE
        rmses = [entry["mean_rmse"] for entry in result.all_results]
        assert all(math.isclose(r, rmses[0], rel_tol=1e-9) for r in rmses)

    def test_sweep_with_test_rank_column(self) -> None:
        rows = _make_sweep_rows()
        # Add war column
        for row in rows:
            row["war"] = row["pa_1"] / 100.0
        cv_splits: list[tuple[list[int], int]] = [([2020], 2021), ([2020, 2021], 2022)]
        result = sweep_cv(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            {},
            {"sample_weight_transform": ["raw", "sqrt"]},
            sample_weight_column="pa_1",
            test_top_n=5,
            test_rank_column="war",
        )
        assert isinstance(result, GridSearchResult)
        assert "sample_weight_transform" in result.best_params

    def test_sweep_with_test_top_n(self) -> None:
        rows = _make_sweep_rows()
        cv_splits: list[tuple[list[int], int]] = [([2020], 2021), ([2020, 2021], 2022)]
        result = sweep_cv(
            rows,
            ["feat_a", "feat_b"],
            ["y"],
            cv_splits,
            {},
            {"sample_weight_transform": ["raw", "sqrt"]},
            sample_weight_column="pa_1",
            test_top_n=5,
        )
        assert isinstance(result, GridSearchResult)
        assert "sample_weight_transform" in result.best_params
