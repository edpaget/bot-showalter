"""Tests for the validation gate pre-flight confidence estimator."""

import random as rng
import statistics

import pytest

from fantasy_baseball_manager.services.validation_gate import (
    PreflightResult,
    PreflightThresholds,
    TargetPreflightDetail,
    preflight_check,
    score_cv_folds,
)


class TestPreflightThresholds:
    def test_defaults(self) -> None:
        t = PreflightThresholds()
        assert t.high_win_rate == 0.75
        assert t.high_target_pct == 0.80
        assert t.medium_win_rate == 0.60
        assert t.medium_target_pct == 0.60

    def test_frozen(self) -> None:
        t = PreflightThresholds()
        with pytest.raises(AttributeError):
            t.high_win_rate = 0.5  # type: ignore[misc]


class TestTargetPreflightDetail:
    def test_creation(self) -> None:
        detail = TargetPreflightDetail(
            target="avg",
            win_rate=0.80,
            mean_delta=-0.01,
            delta_std=0.005,
        )
        assert detail.target == "avg"
        assert detail.win_rate == 0.80
        assert detail.mean_delta == -0.01
        assert detail.delta_std == 0.005


class TestPreflightResult:
    def test_creation(self) -> None:
        detail = TargetPreflightDetail(target="avg", win_rate=0.8, mean_delta=-0.01, delta_std=0.005)
        result = PreflightResult(
            details=(detail,),
            confidence="high",
            recommendation="proceed",
        )
        assert result.confidence == "high"
        assert result.recommendation == "proceed"
        assert len(result.details) == 1


class TestPreflightCheckConsistentImprovement:
    """Candidate consistently beats baseline on most targets across all folds."""

    def test_high_confidence_proceed(self) -> None:
        # 5 folds, 4 targets. Candidate wins every fold on every target.
        baseline = [
            {"avg": 0.050, "obp": 0.060, "slg": 0.080, "woba": 0.055},
            {"avg": 0.052, "obp": 0.062, "slg": 0.082, "woba": 0.057},
            {"avg": 0.048, "obp": 0.058, "slg": 0.078, "woba": 0.053},
            {"avg": 0.051, "obp": 0.061, "slg": 0.081, "woba": 0.056},
            {"avg": 0.049, "obp": 0.059, "slg": 0.079, "woba": 0.054},
        ]
        candidate = [
            {"avg": 0.045, "obp": 0.055, "slg": 0.072, "woba": 0.050},
            {"avg": 0.047, "obp": 0.057, "slg": 0.074, "woba": 0.052},
            {"avg": 0.043, "obp": 0.053, "slg": 0.070, "woba": 0.048},
            {"avg": 0.046, "obp": 0.056, "slg": 0.073, "woba": 0.051},
            {"avg": 0.044, "obp": 0.054, "slg": 0.071, "woba": 0.049},
        ]

        result = preflight_check(candidate, baseline)

        assert result.confidence == "high"
        assert result.recommendation == "proceed"
        assert len(result.details) == 4
        for detail in result.details:
            assert detail.win_rate == 1.0
            assert detail.mean_delta < 0  # candidate RMSE lower

    def test_win_rate_computation(self) -> None:
        # 4 folds, 1 target. Candidate wins 3 of 4.
        baseline = [{"avg": 0.05}, {"avg": 0.06}, {"avg": 0.04}, {"avg": 0.05}]
        candidate = [{"avg": 0.04}, {"avg": 0.05}, {"avg": 0.03}, {"avg": 0.06}]  # loses fold 4

        result = preflight_check(candidate, baseline)

        avg_detail = result.details[0]
        assert avg_detail.target == "avg"
        assert avg_detail.win_rate == 0.75
        # mean_delta = mean of deltas
        deltas = [0.04 - 0.05, 0.05 - 0.06, 0.03 - 0.04, 0.06 - 0.05]
        assert avg_detail.mean_delta == pytest.approx(statistics.mean(deltas))
        assert avg_detail.delta_std == pytest.approx(statistics.stdev(deltas))


class TestPreflightCheckInconsistent:
    """Candidate wins some folds, loses others — mixed results."""

    def test_medium_confidence_marginal(self) -> None:
        # 5 folds, 4 targets. Candidate wins ~60-70% of folds on ~60-70% of targets.
        baseline = [
            {"avg": 0.050, "obp": 0.060, "slg": 0.080, "woba": 0.055},
            {"avg": 0.052, "obp": 0.062, "slg": 0.082, "woba": 0.057},
            {"avg": 0.048, "obp": 0.058, "slg": 0.078, "woba": 0.053},
            {"avg": 0.051, "obp": 0.061, "slg": 0.081, "woba": 0.056},
            {"avg": 0.049, "obp": 0.059, "slg": 0.079, "woba": 0.054},
        ]
        # avg: wins 4/5 (0.8), obp: wins 4/5 (0.8), slg: wins 3/5 (0.6), woba: wins 2/5 (0.4)
        candidate = [
            {"avg": 0.045, "obp": 0.055, "slg": 0.075, "woba": 0.060},
            {"avg": 0.047, "obp": 0.057, "slg": 0.077, "woba": 0.060},
            {"avg": 0.043, "obp": 0.053, "slg": 0.085, "woba": 0.048},
            {"avg": 0.046, "obp": 0.056, "slg": 0.076, "woba": 0.060},
            {"avg": 0.044, "obp": 0.054, "slg": 0.085, "woba": 0.060},
        ]

        result = preflight_check(candidate, baseline)

        # 2 targets (avg, obp) have win_rate >= 0.75 → 50% of targets (below 80% needed for high)
        # 3 targets (avg, obp, slg) have win_rate >= 0.60 → 75% of targets (above 60% needed for medium)
        assert result.confidence == "medium"
        assert result.recommendation == "marginal"


class TestPreflightCheckRegression:
    """Candidate is worse on most folds/targets."""

    def test_low_confidence_skip(self) -> None:
        # 5 folds, 3 targets. Candidate loses most folds on most targets.
        baseline = [
            {"avg": 0.040, "obp": 0.050, "slg": 0.060},
            {"avg": 0.042, "obp": 0.052, "slg": 0.062},
            {"avg": 0.038, "obp": 0.048, "slg": 0.058},
            {"avg": 0.041, "obp": 0.051, "slg": 0.061},
            {"avg": 0.039, "obp": 0.049, "slg": 0.059},
        ]
        candidate = [
            {"avg": 0.055, "obp": 0.065, "slg": 0.075},
            {"avg": 0.057, "obp": 0.067, "slg": 0.077},
            {"avg": 0.053, "obp": 0.063, "slg": 0.073},
            {"avg": 0.056, "obp": 0.066, "slg": 0.076},
            {"avg": 0.054, "obp": 0.064, "slg": 0.074},
        ]

        result = preflight_check(candidate, baseline)

        assert result.confidence == "low"
        assert result.recommendation == "skip"
        for detail in result.details:
            assert detail.win_rate == 0.0
            assert detail.mean_delta > 0  # candidate RMSE higher


class TestPreflightCheckCustomThresholds:
    def test_custom_thresholds(self) -> None:
        # With relaxed thresholds, a borderline scenario becomes "high".
        baseline = [{"avg": 0.050}, {"avg": 0.052}, {"avg": 0.048}, {"avg": 0.051}]
        candidate = [{"avg": 0.045}, {"avg": 0.047}, {"avg": 0.053}, {"avg": 0.046}]
        # win_rate = 3/4 = 0.75; 100% of targets meet 0.75 → high if high_target_pct=1.0... no.
        # By default: 1 target, win_rate=0.75 >= 0.75, 100% targets meet it >= 80% → "high"
        result_default = preflight_check(candidate, baseline)
        assert result_default.confidence == "high"

        # With stricter thresholds:
        strict = PreflightThresholds(high_win_rate=0.90, high_target_pct=1.0)
        result_strict = preflight_check(candidate, baseline, thresholds=strict)
        assert result_strict.confidence != "high"

    def test_edge_case_empty_folds(self) -> None:
        result = preflight_check([], [])
        assert result.confidence == "low"
        assert result.recommendation == "skip"
        assert result.details == ()

    def test_single_fold(self) -> None:
        baseline = [{"avg": 0.050}]
        candidate = [{"avg": 0.040}]
        result = preflight_check(candidate, baseline)
        assert len(result.details) == 1
        assert result.details[0].win_rate == 1.0
        assert result.details[0].delta_std == 0.0  # single fold, no std dev


class TestScoreCvFolds:
    """Tests for the score_cv_folds helper with synthetic data."""

    def _make_rows_by_season(self) -> dict[int, list[dict[str, float]]]:
        """Create synthetic rows grouped by season for testing.

        Each row has features (feat_a, feat_b) and targets (target_avg, target_obp).
        Seasons: 2020, 2021, 2022, 2023, 2024
        """
        rng.seed(42)
        rows_by_season: dict[int, list[dict[str, float]]] = {}
        for season in [2020, 2021, 2022, 2023, 2024]:
            rows: list[dict[str, float]] = []
            for _ in range(50):
                feat_a = rng.gauss(0.260, 0.030)
                feat_b = rng.gauss(0.330, 0.040)
                target_avg = feat_a * 0.6 + feat_b * 0.3 + rng.gauss(0, 0.005)
                target_obp = feat_a * 0.3 + feat_b * 0.6 + rng.gauss(0, 0.005)
                rows.append(
                    {
                        "season": float(season),
                        "feat_a": feat_a,
                        "feat_b": feat_b,
                        "target_avg": target_avg,
                        "target_obp": target_obp,
                    }
                )
            rows_by_season[season] = rows
        return rows_by_season

    def test_returns_per_fold_rmse_dicts(self) -> None:
        rows_by_season = self._make_rows_by_season()
        columns = ["feat_a", "feat_b"]
        targets = ["avg", "obp"]
        seasons = [2020, 2021, 2022, 2023, 2024]
        params: dict[str, int | float] = {"max_iter": 50, "max_depth": 3}

        result = score_cv_folds(columns, targets, rows_by_season, seasons, params)

        # 5 seasons → 3 CV folds (temporal expanding, last season reserved for holdout)
        assert len(result) == 3
        for fold_dict in result:
            assert "avg" in fold_dict
            assert "obp" in fold_dict
            assert fold_dict["avg"] > 0  # RMSE is positive
            assert fold_dict["obp"] > 0

    @pytest.mark.slow
    def test_better_features_yield_lower_rmse(self) -> None:
        """With more relevant features, RMSE should generally be lower."""
        rows_by_season = self._make_rows_by_season()
        targets = ["avg", "obp"]
        seasons = [2020, 2021, 2022, 2023, 2024]
        params: dict[str, int | float] = {"max_iter": 100, "max_depth": 3}

        # Baseline: only feat_a
        baseline_result = score_cv_folds(["feat_a"], targets, rows_by_season, seasons, params)
        # Candidate: both features
        candidate_result = score_cv_folds(["feat_a", "feat_b"], targets, rows_by_season, seasons, params)

        # Check with preflight_check
        pf = preflight_check(candidate_result, baseline_result)
        # Both features should be at least as good as one
        assert pf.confidence in ("high", "medium")
