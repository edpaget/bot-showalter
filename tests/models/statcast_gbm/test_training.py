import math
import random

from fantasy_baseball_manager.models.statcast_gbm.training import (
    compute_permutation_importance,
    extract_features,
    extract_targets,
    fit_models,
    score_predictions,
)


class TestExtractTargets:
    def test_direct_stats(self) -> None:
        rows = [
            {"target_avg": 0.300, "target_obp": 0.380, "target_slg": 0.500, "target_woba": 0.370},
        ]
        result = extract_targets(rows, ["avg", "obp", "slg", "woba"])
        assert result["avg"] == [0.300]
        assert result["obp"] == [0.380]
        assert result["slg"] == [0.500]
        assert result["woba"] == [0.370]

    def test_iso_computed(self) -> None:
        rows = [
            {"target_slg": 0.500, "target_avg": 0.300},
        ]
        result = extract_targets(rows, ["iso"])
        assert len(result["iso"]) == 1
        assert math.isclose(result["iso"][0], 0.200, abs_tol=1e-9)

    def test_babip_computed(self) -> None:
        # babip = (h - hr) / (ab - so - hr + sf)
        rows = [
            {"target_h": 150, "target_hr": 30, "target_ab": 500, "target_so": 100, "target_sf": 5},
        ]
        result = extract_targets(rows, ["babip"])
        expected = (150 - 30) / (500 - 100 - 30 + 5)
        assert len(result["babip"]) == 1
        assert math.isclose(result["babip"][0], expected, abs_tol=1e-9)

    def test_skips_none(self) -> None:
        rows = [
            {"target_avg": 0.300},
            {"target_avg": None},
            {"target_avg": 0.250},
        ]
        result = extract_targets(rows, ["avg"])
        assert result["avg"] == [0.300, 0.250]

    def test_skips_missing_key(self) -> None:
        rows = [
            {"target_avg": 0.300},
            {},  # missing target_avg entirely
        ]
        result = extract_targets(rows, ["avg"])
        assert result["avg"] == [0.300]

    def test_hr_per_9_computed(self) -> None:
        rows = [
            {"target_hr": 20, "target_ip": 180.0},
        ]
        result = extract_targets(rows, ["hr_per_9"])
        expected = 20 * 9 / 180.0
        assert len(result["hr_per_9"]) == 1
        assert math.isclose(result["hr_per_9"][0], expected, abs_tol=1e-9)

    def test_hr_per_9_skips_zero_ip(self) -> None:
        rows = [
            {"target_hr": 5, "target_ip": 0},
        ]
        result = extract_targets(rows, ["hr_per_9"])
        assert result["hr_per_9"] == []

    def test_pitcher_babip_computed(self) -> None:
        # pitcher babip = (h - hr) / (ip * 3 + h - so - hr)
        rows = [
            {"target_h": 150, "target_hr": 20, "target_ip": 180.0, "target_so": 160},
        ]
        result = extract_targets(rows, ["babip"])
        expected = (150 - 20) / (180.0 * 3 + 150 - 160 - 20)
        assert len(result["babip"]) == 1
        assert math.isclose(result["babip"][0], expected, abs_tol=1e-9)

    def test_pitcher_babip_skips_zero_denom(self) -> None:
        # Construct a case where ip*3 + h - so - hr == 0
        rows = [
            {"target_h": 10, "target_hr": 5, "target_ip": 0, "target_so": 5},
        ]
        result = extract_targets(rows, ["babip"])
        assert result["babip"] == []


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


class TestFitModels:
    def test_returns_dict_of_models(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": [0.250, 0.300, 0.275, 0.280]}
        models = fit_models(X, targets, {})
        assert "avg" in models
        assert hasattr(models["avg"], "predict")

    def test_respects_model_params(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": [0.250, 0.300, 0.275, 0.280]}
        models = fit_models(X, targets, {"max_iter": 50})
        assert models["avg"].max_iter == 50


class TestScorePredictions:
    def test_returns_rmse_keys(self) -> None:
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        targets = {"avg": [0.250, 0.300, 0.275, 0.280], "obp": [0.350, 0.400, 0.375, 0.380]}
        models = fit_models(X, targets, {})
        metrics = score_predictions(models, X, targets)
        assert "rmse_avg" in metrics
        assert "rmse_obp" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


class TestComputePermutationImportance:
    def test_returns_all_feature_columns(self) -> None:
        feature_cols = ["feat_a", "feat_b", "feat_c"]
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        targets = {"avg": [0.250, 0.300, 0.275, 0.280]}
        models = fit_models(X, targets, {})
        result = compute_permutation_importance(models, X, targets, feature_cols)
        assert set(result.keys()) == set(feature_cols)

    def test_important_feature_has_positive_impact(self) -> None:
        # feat_a = target (signal); feat_b = noise
        # Use enough data + small min_samples_leaf so the tree actually learns
        gen = random.Random(99)
        all_X = [[float(i), gen.random()] for i in range(80)]
        all_y = [float(i) for i in range(80)]
        X_train = [all_X[i] for i in range(0, 80, 2)]
        y_train = {"y": [all_y[i] for i in range(0, 80, 2)]}
        X_holdout = [all_X[i] for i in range(1, 80, 2)]
        y_holdout = {"y": [all_y[i] for i in range(1, 80, 2)]}
        models = fit_models(X_train, y_train, {"min_samples_leaf": 5})
        result = compute_permutation_importance(models, X_holdout, y_holdout, ["feat_a", "feat_b"])
        assert result["feat_a"] > 0

    def test_irrelevant_feature_has_near_zero_impact(self) -> None:
        gen = random.Random(99)
        all_X = [[float(i), gen.random()] for i in range(80)]
        all_y = [float(i) for i in range(80)]
        X_train = [all_X[i] for i in range(0, 80, 2)]
        y_train = {"y": [all_y[i] for i in range(0, 80, 2)]}
        X_holdout = [all_X[i] for i in range(1, 80, 2)]
        y_holdout = {"y": [all_y[i] for i in range(1, 80, 2)]}
        models = fit_models(X_train, y_train, {"min_samples_leaf": 5})
        result = compute_permutation_importance(models, X_holdout, y_holdout, ["feat_a", "feat_b"])
        assert result["feat_b"] < result["feat_a"]
