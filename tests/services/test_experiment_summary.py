from typing import TYPE_CHECKING

from fantasy_baseball_manager.db.pool import SingleConnectionProvider
from fantasy_baseball_manager.domain.experiment import Experiment, TargetResult
from fantasy_baseball_manager.repos.experiment_repo import SqliteExperimentRepo
from fantasy_baseball_manager.services.experiment_summary import summarize_exploration

if TYPE_CHECKING:
    import sqlite3


def _make_experiment(**overrides: object) -> Experiment:
    defaults: dict[str, object] = {
        "timestamp": "2026-03-01T12:00:00",
        "hypothesis": "test hypothesis",
        "model": "statcast-gbm-preseason",
        "player_type": "batter",
        "feature_diff": {"added": ["barrel_rate"], "removed": []},
        "seasons": {"train": [2021, 2022, 2023], "holdout": [2024]},
        "params": {"n_estimators": 500},
        "target_results": {
            "slg": TargetResult(rmse=0.082, baseline_rmse=0.085, delta=-0.003, delta_pct=-3.53),
        },
        "conclusion": "conclusion",
    }
    defaults.update(overrides)
    return Experiment(**defaults)  # type: ignore[arg-type]


class TestSummarizeExploration:
    def test_total_experiments_count(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(SingleConnectionProvider(conn))
        repo.save(_make_experiment())
        repo.save(_make_experiment())
        repo.save(_make_experiment(model="other-model"))

        summary = summarize_exploration(repo, "statcast-gbm-preseason", "batter")
        assert summary.total_experiments == 2

    def test_features_tested_with_best_result(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(SingleConnectionProvider(conn))
        # Experiment 1: barrel_rate with avg delta_pct -3.53
        repo.save(
            _make_experiment(
                feature_diff={"added": ["barrel_rate"], "removed": []},
                target_results={
                    "slg": TargetResult(rmse=0.082, baseline_rmse=0.085, delta=-0.003, delta_pct=-3.53),
                },
            )
        )
        # Experiment 2: barrel_rate with avg delta_pct -5.0 (better)
        repo.save(
            _make_experiment(
                feature_diff={"added": ["barrel_rate"], "removed": []},
                target_results={
                    "slg": TargetResult(rmse=0.080, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.0),
                },
            )
        )
        # Experiment 3: sprint_speed
        repo.save(
            _make_experiment(
                feature_diff={"added": ["sprint_speed"], "removed": []},
                target_results={
                    "slg": TargetResult(rmse=0.084, baseline_rmse=0.085, delta=-0.001, delta_pct=-1.0),
                },
            )
        )

        summary = summarize_exploration(repo, "statcast-gbm-preseason", "batter")
        assert len(summary.features_tested) == 2

        # Features sorted by best_delta_pct ascending
        barrel = next(f for f in summary.features_tested if f.feature == "barrel_rate")
        assert barrel.times_tested == 2
        assert barrel.best_delta_pct == -5.0

        sprint = next(f for f in summary.features_tested if f.feature == "sprint_speed")
        assert sprint.times_tested == 1
        assert sprint.best_delta_pct == -1.0

    def test_targets_explored_with_best_rmse(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(SingleConnectionProvider(conn))
        repo.save(
            _make_experiment(
                target_results={
                    "slg": TargetResult(rmse=0.082, baseline_rmse=0.085, delta=-0.003, delta_pct=-3.53),
                    "obp": TargetResult(rmse=0.040, baseline_rmse=0.042, delta=-0.002, delta_pct=-4.76),
                },
            )
        )
        repo.save(
            _make_experiment(
                target_results={
                    "slg": TargetResult(rmse=0.080, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.88),
                },
            )
        )

        summary = summarize_exploration(repo, "statcast-gbm-preseason", "batter")
        assert len(summary.targets_explored) == 2

        slg = next(t for t in summary.targets_explored if t.target == "slg")
        assert slg.experiments_count == 2
        assert slg.best_rmse == 0.080
        assert slg.best_delta_pct == -5.88

        obp = next(t for t in summary.targets_explored if t.target == "obp")
        assert obp.experiments_count == 1

    def test_overall_best_experiment(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(SingleConnectionProvider(conn))
        # Experiment with avg delta_pct = -3.53
        repo.save(
            _make_experiment(
                target_results={
                    "slg": TargetResult(rmse=0.082, baseline_rmse=0.085, delta=-0.003, delta_pct=-3.53),
                },
            )
        )
        # Experiment with avg delta_pct = -5.0 (better)
        id2 = repo.save(
            _make_experiment(
                target_results={
                    "slg": TargetResult(rmse=0.080, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.0),
                },
            )
        )

        summary = summarize_exploration(repo, "statcast-gbm-preseason", "batter")
        assert summary.best_experiment_id == id2
        assert summary.best_experiment_delta_pct == -5.0

    def test_empty_summary(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(SingleConnectionProvider(conn))

        summary = summarize_exploration(repo, "nonexistent-model", "batter")
        assert summary.total_experiments == 0
        assert summary.features_tested == []
        assert summary.targets_explored == []
        assert summary.best_experiment_id is None
        assert summary.best_experiment_delta_pct is None

    def test_filters_by_model_and_player_type(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(SingleConnectionProvider(conn))
        repo.save(_make_experiment(model="model-a", player_type="batter"))
        repo.save(_make_experiment(model="model-a", player_type="pitcher"))
        repo.save(_make_experiment(model="model-b", player_type="batter"))

        summary = summarize_exploration(repo, "model-a", "batter")
        assert summary.total_experiments == 1
        assert summary.model == "model-a"
        assert summary.player_type == "batter"

    def test_features_from_removed_are_counted(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(SingleConnectionProvider(conn))
        repo.save(
            _make_experiment(
                feature_diff={"added": [], "removed": ["barrel_rate"]},
                target_results={
                    "slg": TargetResult(rmse=0.084, baseline_rmse=0.085, delta=-0.001, delta_pct=-1.0),
                },
            )
        )

        summary = summarize_exploration(repo, "statcast-gbm-preseason", "batter")
        assert len(summary.features_tested) == 1
        assert summary.features_tested[0].feature == "barrel_rate"
