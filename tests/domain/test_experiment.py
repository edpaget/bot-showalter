from fantasy_baseball_manager.domain.experiment import (
    Experiment,
    ExplorationSummary,
    FeatureExplorationResult,
    TargetExplorationResult,
    TargetResult,
)
from fantasy_baseball_manager.domain.identity import PlayerType


class TestTargetResult:
    def test_fields_accessible(self) -> None:
        tr = TargetResult(rmse=0.082, baseline_rmse=0.085, delta=-0.003, delta_pct=-3.53)
        assert tr.rmse == 0.082
        assert tr.baseline_rmse == 0.085
        assert tr.delta == -0.003
        assert tr.delta_pct == -3.53

    def test_frozen(self) -> None:
        tr = TargetResult(rmse=0.1, baseline_rmse=0.2, delta=-0.1, delta_pct=-50.0)
        try:
            tr.rmse = 0.5  # type: ignore[misc]
            raise AssertionError("Should have raised")  # noqa: TRY301
        except AttributeError:
            pass


class TestExperiment:
    def test_all_fields_accessible(self) -> None:
        exp = Experiment(
            timestamp="2026-03-01T12:00:00",
            hypothesis="adding barrel rate will improve SLG",
            model="statcast-gbm-preseason",
            player_type=PlayerType.BATTER,
            feature_diff={"added": ["barrel_rate"], "removed": []},
            seasons={"train": [2021, 2022, 2023], "holdout": [2024]},
            params={"n_estimators": 500, "learning_rate": 0.05},
            target_results={
                "slg": TargetResult(rmse=0.082, baseline_rmse=0.085, delta=-0.003, delta_pct=-3.53),
            },
            conclusion="barrel rate improved SLG prediction",
            tags=["feature", "batter"],
            parent_id=1,
            id=42,
        )
        assert exp.timestamp == "2026-03-01T12:00:00"
        assert exp.hypothesis == "adding barrel rate will improve SLG"
        assert exp.model == "statcast-gbm-preseason"
        assert exp.player_type == "batter"
        assert exp.feature_diff == {"added": ["barrel_rate"], "removed": []}
        assert exp.seasons == {"train": [2021, 2022, 2023], "holdout": [2024]}
        assert exp.params == {"n_estimators": 500, "learning_rate": 0.05}
        assert exp.target_results["slg"].rmse == 0.082
        assert exp.conclusion == "barrel rate improved SLG prediction"
        assert exp.tags == ["feature", "batter"]
        assert exp.parent_id == 1
        assert exp.id == 42

    def test_defaults(self) -> None:
        exp = Experiment(
            timestamp="2026-03-01T12:00:00",
            hypothesis="test",
            model="test-model",
            player_type=PlayerType.BATTER,
            feature_diff={"added": [], "removed": []},
            seasons={"train": [2023], "holdout": [2024]},
            params={},
            target_results={},
            conclusion="n/a",
        )
        assert exp.tags == []
        assert exp.parent_id is None
        assert exp.id is None

    def test_frozen(self) -> None:
        exp = Experiment(
            timestamp="2026-03-01T12:00:00",
            hypothesis="test",
            model="test-model",
            player_type=PlayerType.BATTER,
            feature_diff={"added": [], "removed": []},
            seasons={"train": [2023], "holdout": [2024]},
            params={},
            target_results={},
            conclusion="n/a",
        )
        try:
            exp.model = "other"  # type: ignore[misc]
            raise AssertionError("Should have raised")  # noqa: TRY301
        except AttributeError:
            pass


class TestFeatureExplorationResult:
    def test_fields_accessible(self) -> None:
        r = FeatureExplorationResult(
            feature="barrel_rate",
            best_delta_pct=-3.53,
            best_experiment_id=42,
            times_tested=5,
        )
        assert r.feature == "barrel_rate"
        assert r.best_delta_pct == -3.53
        assert r.best_experiment_id == 42
        assert r.times_tested == 5

    def test_frozen(self) -> None:
        r = FeatureExplorationResult(
            feature="barrel_rate",
            best_delta_pct=-3.53,
            best_experiment_id=42,
            times_tested=5,
        )
        try:
            r.feature = "other"  # type: ignore[misc]
            raise AssertionError("Should have raised")  # noqa: TRY301
        except AttributeError:
            pass


class TestTargetExplorationResult:
    def test_fields_accessible(self) -> None:
        r = TargetExplorationResult(
            target="slg",
            best_rmse=0.082,
            best_delta_pct=-3.53,
            best_experiment_id=42,
            experiments_count=10,
        )
        assert r.target == "slg"
        assert r.best_rmse == 0.082
        assert r.best_delta_pct == -3.53
        assert r.best_experiment_id == 42
        assert r.experiments_count == 10

    def test_frozen(self) -> None:
        r = TargetExplorationResult(
            target="slg",
            best_rmse=0.082,
            best_delta_pct=-3.53,
            best_experiment_id=42,
            experiments_count=10,
        )
        try:
            r.target = "other"  # type: ignore[misc]
            raise AssertionError("Should have raised")  # noqa: TRY301
        except AttributeError:
            pass


class TestExplorationSummary:
    def test_fields_accessible(self) -> None:
        feat = FeatureExplorationResult(
            feature="barrel_rate", best_delta_pct=-3.53, best_experiment_id=1, times_tested=5
        )
        tgt = TargetExplorationResult(
            target="slg", best_rmse=0.082, best_delta_pct=-3.53, best_experiment_id=1, experiments_count=10
        )
        s = ExplorationSummary(
            model="statcast-gbm-preseason",
            player_type=PlayerType.BATTER,
            total_experiments=10,
            features_tested=[feat],
            targets_explored=[tgt],
            best_experiment_id=1,
            best_experiment_delta_pct=-3.53,
        )
        assert s.model == "statcast-gbm-preseason"
        assert s.player_type == "batter"
        assert s.total_experiments == 10
        assert len(s.features_tested) == 1
        assert s.features_tested[0].feature == "barrel_rate"
        assert len(s.targets_explored) == 1
        assert s.targets_explored[0].target == "slg"
        assert s.best_experiment_id == 1
        assert s.best_experiment_delta_pct == -3.53

    def test_none_defaults(self) -> None:
        s = ExplorationSummary(
            model="m",
            player_type=PlayerType.BATTER,
            total_experiments=0,
            features_tested=[],
            targets_explored=[],
            best_experiment_id=None,
            best_experiment_delta_pct=None,
        )
        assert s.best_experiment_id is None
        assert s.best_experiment_delta_pct is None

    def test_frozen(self) -> None:
        s = ExplorationSummary(
            model="m",
            player_type=PlayerType.BATTER,
            total_experiments=0,
            features_tested=[],
            targets_explored=[],
            best_experiment_id=None,
            best_experiment_delta_pct=None,
        )
        try:
            s.model = "other"  # type: ignore[misc]
            raise AssertionError("Should have raised")  # noqa: TRY301
        except AttributeError:
            pass
