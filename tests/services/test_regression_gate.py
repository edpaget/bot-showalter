from dataclasses import dataclass, field
from typing import Any

from fantasy_baseball_manager.domain.evaluation import (
    ComparisonResult,
    StatMetrics,
    SystemMetrics,
)
from fantasy_baseball_manager.models import ModelConfig, PredictResult, TrainResult
from fantasy_baseball_manager.services.regression_gate import (
    GateConfig,
    RegressionGateRunner,
)


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
    baseline_version: str = "latest",
    candidate_system: str = "test-model",
    candidate_version: str = "gate-h2024",
    season: int = 2024,
    candidate_wins: bool = True,
) -> ComparisonResult:
    """Build a ComparisonResult where the candidate either wins or loses on all stats."""
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
        # Default: candidate wins
        return _make_comparison(season=season, candidate_version=f"gate-h{season}")


@dataclass
class FakeProjectionRepo:
    upserted: list[Any] = field(default_factory=list)
    deleted: list[tuple[str, str]] = field(default_factory=list)

    def upsert(self, projection: Any) -> int:
        self.upserted.append(projection)
        return len(self.upserted)

    def get_by_player_season(
        self, player_id: int, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Any]:
        return []

    def get_by_season(
        self, season: int, system: str | None = None, *, include_distributions: bool = False
    ) -> list[Any]:
        return []

    def get_by_system_version(self, system: str, version: str) -> list[Any]:
        return []

    def delete_by_system_version(self, system: str, version: str) -> int:
        self.deleted.append((system, version))
        return 1

    def upsert_distributions(self, projection_id: int, distributions: list[Any]) -> None:
        pass

    def get_distributions(self, projection_id: int) -> list[Any]:
        return []


def _make_gate_config(
    holdout_seasons: list[int] | None = None,
    top: int | None = None,
) -> GateConfig:
    return GateConfig(
        model_name="test-model",
        base_training_seasons=[2021, 2022, 2023],
        holdout_seasons=holdout_seasons or [2024, 2025],
        baseline_system="test-model",
        baseline_version="latest",
        top=top,
        model_params={},
        data_dir="./data",
        artifacts_dir="./artifacts",
    )


def _make_runner(
    model: FakeModel | None = None,
    evaluator: FakeEvaluator | None = None,
    projection_repo: FakeProjectionRepo | None = None,
) -> tuple[RegressionGateRunner, FakeModel, FakeEvaluator, FakeProjectionRepo]:
    m = model or FakeModel()
    e = evaluator or FakeEvaluator()
    p = projection_repo or FakeProjectionRepo()
    runner = RegressionGateRunner(model=m, evaluator=e, projection_repo=p)
    return runner, m, e, p


class TestGateRunner:
    def test_passes_when_all_checks_pass(self) -> None:
        runner, _model, _eval, _repo = _make_runner()
        config = _make_gate_config()
        result = runner.run(config)
        assert result.passed
        assert result.model_name == "test-model"
        assert result.baseline == "test-model/latest"

    def test_fails_when_any_check_fails(self) -> None:
        evaluator = FakeEvaluator()
        # 2024 passes, 2025 fails
        evaluator.set_result(
            2024,
            None,
            _make_comparison(season=2024, candidate_version="gate-h2024", candidate_wins=True),
        )
        evaluator.set_result(
            2025,
            None,
            _make_comparison(season=2025, candidate_version="gate-h2025", candidate_wins=False),
        )
        runner, _model, _eval, _repo = _make_runner(evaluator=evaluator)
        config = _make_gate_config()
        result = runner.run(config)
        assert not result.passed

    def test_runs_top_and_full_when_top_specified(self) -> None:
        runner, _model, evaluator, _repo = _make_runner()
        config = _make_gate_config(top=300)
        result = runner.run(config)
        # 2 holdouts × 2 segments (full + top-300) = 4
        assert len(result.segments) == 4
        segment_labels = [(s.season, s.segment) for s in result.segments]
        assert (2024, "full") in segment_labels
        assert (2024, "top-300") in segment_labels
        assert (2025, "full") in segment_labels
        assert (2025, "top-300") in segment_labels

    def test_runs_only_full_when_no_top(self) -> None:
        runner, _model, _eval, _repo = _make_runner()
        config = _make_gate_config(top=None)
        result = runner.run(config)
        # 2 holdouts × 1 segment (full only) = 2
        assert len(result.segments) == 2
        assert all(s.segment == "full" for s in result.segments)

    def test_training_seasons_append_holdout(self) -> None:
        runner, model, _eval, _repo = _make_runner()
        config = _make_gate_config(holdout_seasons=[2024])
        runner.run(config)
        assert len(model.train_calls) == 1
        assert model.train_calls[0].seasons == [2021, 2022, 2023, 2024]

    def test_predict_uses_holdout_only(self) -> None:
        runner, model, _eval, _repo = _make_runner()
        config = _make_gate_config(holdout_seasons=[2024])
        runner.run(config)
        assert len(model.predict_calls) == 1
        assert model.predict_calls[0].seasons == [2024]

    def test_candidate_version_format(self) -> None:
        runner, model, _eval, _repo = _make_runner()
        config = _make_gate_config(holdout_seasons=[2024])
        runner.run(config)
        assert model.train_calls[0].version == "gate-h2024"
        assert model.predict_calls[0].version == "gate-h2024"

    def test_cleanup_deletes_gate_predictions(self) -> None:
        runner, _model, _eval, repo = _make_runner()
        config = _make_gate_config(holdout_seasons=[2024, 2025])
        runner.cleanup(config)
        assert ("test-model", "gate-h2024") in repo.deleted
        assert ("test-model", "gate-h2025") in repo.deleted
