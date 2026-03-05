"""Tests for the validation gate pre-flight confidence estimator and full orchestrator."""

import random as rng
import statistics
from dataclasses import dataclass, field
from typing import Any

import pytest

from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    RegressionCheckResult,
    StatMetrics,
    SystemMetrics,
)
from fantasy_baseball_manager.models import ModelConfig, PredictResult, TrainResult
from fantasy_baseball_manager.models.gbm_training_backend import GBMTrainingBackend
from fantasy_baseball_manager.services.validation_gate import (
    FullValidationConfig,
    FullValidationRunner,
    PreflightResult,
    PreflightThresholds,
    TargetPreflightDetail,
    ValidationResult,
    ValidationSegmentResult,
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

        result = score_cv_folds(columns, targets, rows_by_season, seasons, params, GBMTrainingBackend())

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
        backend = GBMTrainingBackend()
        baseline_result = score_cv_folds(["feat_a"], targets, rows_by_season, seasons, params, backend)
        # Candidate: both features
        candidate_result = score_cv_folds(["feat_a", "feat_b"], targets, rows_by_season, seasons, params, backend)

        # Check with preflight_check
        pf = preflight_check(candidate_result, baseline_result)
        # Both features should be at least as good as one
        assert pf.confidence in ("high", "medium")


# ---------------------------------------------------------------------------
# Fakes for FullValidationRunner tests
# ---------------------------------------------------------------------------


def _make_stat_metrics(rmse: float = 1.0, rank_correlation: float = 0.9) -> StatMetrics:
    return StatMetrics(
        rmse=rmse,
        mae=rmse * 0.8,
        correlation=rank_correlation,
        rank_correlation=rank_correlation,
        r_squared=rank_correlation,
        mean_error=0.0,
        n=100,
    )


def _make_comparison(
    *,
    baseline_system: str = "test-model",
    baseline_version: str = "old-h2024",
    candidate_system: str = "test-model",
    candidate_version: str = "new-h2024",
    season: int = 2024,
    candidate_wins: bool = True,
) -> ComparisonResult:
    if candidate_wins:
        baseline_metrics = {"hr": _make_stat_metrics(rmse=2.0, rank_correlation=0.8)}
        candidate_metrics = {"hr": _make_stat_metrics(rmse=1.0, rank_correlation=0.9)}
    else:
        baseline_metrics = {"hr": _make_stat_metrics(rmse=1.0, rank_correlation=0.9)}
        candidate_metrics = {"hr": _make_stat_metrics(rmse=2.0, rank_correlation=0.8)}

    return ComparisonResult(
        season=season,
        stats=["hr"],
        systems=[
            SystemMetrics(
                system=baseline_system,
                version=baseline_version,
                source_type="first_party",
                metrics=baseline_metrics,
            ),
            SystemMetrics(
                system=candidate_system,
                version=candidate_version,
                source_type="first_party",
                metrics=candidate_metrics,
            ),
        ],
    )


@dataclass
class FakeModel:
    name: str = "test-model"
    description: str = "A test model"
    supported_operations: frozenset[str] = frozenset({"train", "predict"})
    artifact_type: str = "model"
    train_calls: list[ModelConfig] = field(default_factory=list)
    predict_calls: list[ModelConfig] = field(default_factory=list)
    predict_result: PredictResult = field(
        default_factory=lambda: PredictResult(
            model_name="test-model",
            predictions=[
                {"player_id": 1, "season": 2024, "player_type": "batter", "hr": 30},
            ],
            output_path="out.csv",
        )
    )

    def train(self, config: ModelConfig) -> TrainResult:
        self.train_calls.append(config)
        return TrainResult(model_name=self.name, metrics={"rmse": 1.0}, artifacts_path="artifacts/")

    def predict(self, config: ModelConfig) -> PredictResult:
        self.predict_calls.append(config)
        return self.predict_result


@dataclass
class FakeEvaluator:
    compare_calls: list[tuple[list[tuple[str, str]], int, int | None]] = field(default_factory=list)
    _results: dict[tuple[int, int | None], ComparisonResult] = field(default_factory=dict)

    def set_result(self, season: int, top: int | None, result: ComparisonResult) -> None:
        self._results[(season, top)] = result

    def compare(
        self,
        systems: list[tuple[str, str]],
        season: int,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        top: int | None = None,
        **kwargs: Any,
    ) -> ComparisonResult:
        self.compare_calls.append((systems, season, top))
        result = self._results.get((season, top))
        if result is not None:
            return result
        return _make_comparison(season=season, candidate_wins=True)


@dataclass
class FakeProjectionRepo:
    upserted: list[Any] = field(default_factory=list)
    deleted: list[tuple[str, str]] = field(default_factory=list)
    _existing: dict[tuple[str, str], list[Any]] = field(default_factory=dict)

    def upsert(self, projection: Any) -> int:
        self.upserted.append(projection)
        return len(self.upserted)

    def get_by_system_version(self, system: str, version: str) -> list[Any]:
        return self._existing.get((system, version), [])

    def delete_by_system_version(self, system: str, version: str) -> int:
        self.deleted.append((system, version))
        return 1

    def upsert_distributions(self, projection_id: int, distributions: list[Any]) -> None:
        pass

    def set_existing(self, system: str, version: str, data: list[Any]) -> None:
        self._existing[(system, version)] = data


def _make_validation_config(
    holdout_seasons: list[int] | None = None,
    train_seasons: list[int] | None = None,
    top: int | None = None,
    old_params: dict[str, Any] | None = None,
    new_params: dict[str, Any] | None = None,
) -> FullValidationConfig:
    return FullValidationConfig(
        model_name="test-model",
        old_version="old",
        new_version="new",
        old_params=old_params or {},
        new_params=new_params or {},
        holdout_seasons=holdout_seasons or [2023, 2024],
        train_seasons=train_seasons or [2019, 2020, 2021, 2022, 2023, 2024],
        top=top,
        data_dir="./data",
        artifacts_dir="./artifacts",
    )


def _make_validation_runner(
    model: FakeModel | None = None,
    evaluator: FakeEvaluator | None = None,
    projection_repo: FakeProjectionRepo | None = None,
) -> tuple[FullValidationRunner, FakeModel, FakeEvaluator, FakeProjectionRepo]:
    m = model or FakeModel()
    e = evaluator or FakeEvaluator()
    p = projection_repo or FakeProjectionRepo()
    runner = FullValidationRunner(model=m, evaluator=e, projection_repo=p)
    return runner, m, e, p


# ---------------------------------------------------------------------------
# Domain type tests
# ---------------------------------------------------------------------------


class TestFullValidationConfig:
    def test_frozen(self) -> None:
        config = _make_validation_config()
        with pytest.raises(AttributeError):
            config.model_name = "other"  # type: ignore[misc]


class TestValidationSegmentResult:
    def test_creation(self) -> None:
        check = RegressionCheckResult(passed=True, rmse_passed=True, rank_correlation_passed=True, explanation="ok")
        seg = ValidationSegmentResult(season=2024, segment="full", check=check)
        assert seg.season == 2024
        assert seg.segment == "full"
        assert seg.check.passed


class TestValidationResult:
    def test_passed_when_all_segments_pass(self) -> None:
        check = RegressionCheckResult(passed=True, rmse_passed=True, rank_correlation_passed=True, explanation="ok")
        result = ValidationResult(
            model_name="m",
            old_version="v1",
            new_version="v2",
            segments=[ValidationSegmentResult(season=2024, segment="full", check=check)],
        )
        assert result.passed

    def test_fails_when_any_segment_fails(self) -> None:
        pass_check = RegressionCheckResult(
            passed=True, rmse_passed=True, rank_correlation_passed=True, explanation="ok"
        )
        fail_check = RegressionCheckResult(
            passed=False, rmse_passed=False, rank_correlation_passed=True, explanation="rmse regressed"
        )
        result = ValidationResult(
            model_name="m",
            old_version="v1",
            new_version="v2",
            segments=[
                ValidationSegmentResult(season=2023, segment="full", check=pass_check),
                ValidationSegmentResult(season=2024, segment="full", check=fail_check),
            ],
        )
        assert not result.passed

    def test_preflight_stored(self) -> None:
        preflight = PreflightResult(details=(), confidence="high", recommendation="proceed")
        result = ValidationResult(
            model_name="m",
            old_version="v1",
            new_version="v2",
            segments=[],
            preflight=preflight,
        )
        assert result.preflight is not None
        assert result.preflight.confidence == "high"


# ---------------------------------------------------------------------------
# FullValidationRunner tests
# ---------------------------------------------------------------------------


class TestFullValidationRunner:
    def test_passes_when_all_checks_pass(self) -> None:
        runner, _model, _eval, _repo = _make_validation_runner()
        config = _make_validation_config()
        result = runner.run(config)
        assert result.passed
        assert result.model_name == "test-model"
        assert result.old_version == "old"
        assert result.new_version == "new"

    def test_fails_when_any_segment_fails(self) -> None:
        evaluator = FakeEvaluator()
        evaluator.set_result(
            2023,
            None,
            _make_comparison(season=2023, candidate_wins=True),
        )
        evaluator.set_result(
            2024,
            None,
            _make_comparison(season=2024, candidate_wins=False),
        )
        runner, _model, _eval, _repo = _make_validation_runner(evaluator=evaluator)
        config = _make_validation_config()
        result = runner.run(config)
        assert not result.passed

    def test_runs_full_and_top_when_top_specified(self) -> None:
        runner, _model, evaluator, _repo = _make_validation_runner()
        config = _make_validation_config(top=300)
        result = runner.run(config)
        # 2 holdouts × 2 segments (full + top-300) = 4
        assert len(result.segments) == 4
        segment_labels = [(s.season, s.segment) for s in result.segments]
        assert (2023, "full") in segment_labels
        assert (2023, "top-300") in segment_labels
        assert (2024, "full") in segment_labels
        assert (2024, "top-300") in segment_labels

    def test_runs_only_full_when_no_top(self) -> None:
        runner, _model, _eval, _repo = _make_validation_runner()
        config = _make_validation_config(top=None)
        result = runner.run(config)
        assert len(result.segments) == 2
        assert all(s.segment == "full" for s in result.segments)

    def test_training_seasons_exclude_holdout(self) -> None:
        runner, model, _eval, _repo = _make_validation_runner()
        config = _make_validation_config(
            holdout_seasons=[2023],
            train_seasons=[2019, 2020, 2021, 2022, 2023, 2024],
        )
        runner.run(config)
        # Old and new each trained once → 2 train calls
        assert len(model.train_calls) == 2
        for call in model.train_calls:
            assert 2023 not in call.seasons
            assert call.seasons == [2019, 2020, 2021, 2022, 2024]

    def test_version_tag_format(self) -> None:
        runner, model, _eval, _repo = _make_validation_runner()
        config = _make_validation_config(holdout_seasons=[2024])
        runner.run(config)
        old_versions = [c.version for c in model.train_calls if c.version and c.version.startswith("old")]
        new_versions = [c.version for c in model.train_calls if c.version and c.version.startswith("new")]
        assert "old-h2024" in old_versions
        assert "new-h2024" in new_versions

    def test_reuses_existing_predictions(self) -> None:
        repo = FakeProjectionRepo()
        repo.set_existing("test-model", "old-h2024", [{"player_id": 1, "hr": 30}])
        runner, model, _eval, _repo = _make_validation_runner(projection_repo=repo)
        config = _make_validation_config(holdout_seasons=[2024])
        runner.run(config)
        # Old version has existing predictions → skip train+predict for old
        # New version does not → train+predict for new
        old_train_calls = [c for c in model.train_calls if c.version == "old-h2024"]
        new_train_calls = [c for c in model.train_calls if c.version == "new-h2024"]
        assert len(old_train_calls) == 0
        assert len(new_train_calls) == 1

    def test_cleanup_deletes_all_version_tags(self) -> None:
        runner, _model, _eval, repo = _make_validation_runner()
        config = _make_validation_config(holdout_seasons=[2023, 2024])
        runner.cleanup(config)
        assert ("test-model", "old-h2023") in repo.deleted
        assert ("test-model", "new-h2023") in repo.deleted
        assert ("test-model", "old-h2024") in repo.deleted
        assert ("test-model", "new-h2024") in repo.deleted

    def test_preflight_passed_through(self) -> None:
        runner, _model, _eval, _repo = _make_validation_runner()
        config = _make_validation_config()
        preflight = PreflightResult(details=(), confidence="high", recommendation="proceed")
        result = runner.run(config, preflight=preflight)
        assert result.preflight is preflight

    def test_old_and_new_trained_with_correct_params(self) -> None:
        runner, model, _eval, _repo = _make_validation_runner()
        config = _make_validation_config(
            holdout_seasons=[2024],
            old_params={"max_depth": 3},
            new_params={"max_depth": 5},
        )
        runner.run(config)
        old_calls = [c for c in model.train_calls if c.version == "old-h2024"]
        new_calls = [c for c in model.train_calls if c.version == "new-h2024"]
        assert len(old_calls) == 1
        assert len(new_calls) == 1
        assert old_calls[0].model_params == {"max_depth": 3}
        assert new_calls[0].model_params == {"max_depth": 5}
