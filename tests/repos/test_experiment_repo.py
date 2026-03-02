from typing import TYPE_CHECKING

from fantasy_baseball_manager.domain.experiment import Experiment, TargetResult
from fantasy_baseball_manager.repos.experiment_repo import ExperimentFilter, SqliteExperimentRepo

if TYPE_CHECKING:
    import sqlite3


def _make_experiment(**overrides: object) -> Experiment:
    defaults: dict[str, object] = {
        "timestamp": "2026-03-01T12:00:00",
        "hypothesis": "adding barrel rate improves SLG",
        "model": "statcast-gbm-preseason",
        "player_type": "batter",
        "feature_diff": {"added": ["barrel_rate"], "removed": []},
        "seasons": {"train": [2021, 2022, 2023], "holdout": [2024]},
        "params": {"n_estimators": 500},
        "target_results": {
            "slg": TargetResult(rmse=0.082, baseline_rmse=0.085, delta=-0.003, delta_pct=-3.53),
        },
        "conclusion": "barrel rate improved SLG prediction",
    }
    defaults.update(overrides)
    return Experiment(**defaults)  # type: ignore[arg-type]


class TestSqliteExperimentRepo:
    def test_save_and_get_round_trip(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        exp = _make_experiment(
            tags=["feature", "batter"],
            parent_id=None,
        )
        exp_id = repo.save(exp)
        assert exp_id > 0

        result = repo.get(exp_id)
        assert result is not None
        assert result.id == exp_id
        assert result.timestamp == "2026-03-01T12:00:00"
        assert result.hypothesis == "adding barrel rate improves SLG"
        assert result.model == "statcast-gbm-preseason"
        assert result.player_type == "batter"
        assert result.feature_diff == {"added": ["barrel_rate"], "removed": []}
        assert result.seasons == {"train": [2021, 2022, 2023], "holdout": [2024]}
        assert result.params == {"n_estimators": 500}
        assert result.target_results["slg"].rmse == 0.082
        assert result.target_results["slg"].baseline_rmse == 0.085
        assert result.target_results["slg"].delta == -0.003
        assert result.target_results["slg"].delta_pct == -3.53
        assert result.conclusion == "barrel rate improved SLG prediction"
        assert result.tags == ["feature", "batter"]
        assert result.parent_id is None

    def test_get_returns_none_when_missing(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        assert repo.get(9999) is None

    def test_list_all(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(timestamp="2026-03-01T10:00:00"))
        repo.save(_make_experiment(timestamp="2026-03-01T12:00:00"))
        repo.save(_make_experiment(timestamp="2026-03-01T11:00:00"))

        results = repo.list()
        assert len(results) == 3
        # Ordered by timestamp DESC
        assert results[0].timestamp == "2026-03-01T12:00:00"
        assert results[1].timestamp == "2026-03-01T11:00:00"
        assert results[2].timestamp == "2026-03-01T10:00:00"

    def test_list_filter_by_model(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(model="model-a"))
        repo.save(_make_experiment(model="model-b"))
        repo.save(_make_experiment(model="model-a"))

        results = repo.list(ExperimentFilter(model="model-a"))
        assert len(results) == 2
        assert all(r.model == "model-a" for r in results)

    def test_list_filter_by_player_type(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(player_type="batter"))
        repo.save(_make_experiment(player_type="pitcher"))

        results = repo.list(ExperimentFilter(player_type="pitcher"))
        assert len(results) == 1
        assert results[0].player_type == "pitcher"

    def test_list_filter_by_tag(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(tags=["feature", "batter"]))
        repo.save(_make_experiment(tags=["hyperparameter"]))
        repo.save(_make_experiment(tags=["feature", "pitcher"]))

        results = repo.list(ExperimentFilter(tag="feature"))
        assert len(results) == 2

    def test_list_filter_by_date_range(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(timestamp="2026-01-01T00:00:00"))
        repo.save(_make_experiment(timestamp="2026-02-01T00:00:00"))
        repo.save(_make_experiment(timestamp="2026-03-01T00:00:00"))

        results = repo.list(ExperimentFilter(since="2026-01-15", until="2026-02-15"))
        assert len(results) == 1
        assert results[0].timestamp == "2026-02-01T00:00:00"

    def test_list_combined_filters(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(model="model-a", player_type="batter", tags=["feature"]))
        repo.save(_make_experiment(model="model-a", player_type="pitcher", tags=["feature"]))
        repo.save(_make_experiment(model="model-b", player_type="batter", tags=["feature"]))

        results = repo.list(ExperimentFilter(model="model-a", player_type="batter"))
        assert len(results) == 1
        assert results[0].model == "model-a"
        assert results[0].player_type == "batter"

    def test_list_filter_by_parent_id(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        parent_id = repo.save(_make_experiment())
        repo.save(_make_experiment(parent_id=parent_id))
        repo.save(_make_experiment(parent_id=parent_id))
        repo.save(_make_experiment())

        results = repo.list(ExperimentFilter(parent_id=parent_id))
        assert len(results) == 2
        assert all(r.parent_id == parent_id for r in results)

    def test_delete(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        exp_id = repo.save(_make_experiment())
        assert repo.get(exp_id) is not None

        repo.delete(exp_id)
        assert repo.get(exp_id) is None

    def test_delete_nonexistent_is_noop(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.delete(9999)  # should not raise

    def test_parent_child_relationship(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        parent_id = repo.save(_make_experiment(hypothesis="parent experiment"))
        child_id = repo.save(_make_experiment(hypothesis="child experiment", parent_id=parent_id))

        parent = repo.get(parent_id)
        child = repo.get(child_id)
        assert parent is not None
        assert child is not None
        assert child.parent_id == parent_id
        assert parent.parent_id is None

    def test_empty_tags_round_trip(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        exp_id = repo.save(_make_experiment(tags=[]))

        result = repo.get(exp_id)
        assert result is not None
        assert result.tags == []

    def test_multiple_target_results_round_trip(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        target_results = {
            "slg": TargetResult(rmse=0.082, baseline_rmse=0.085, delta=-0.003, delta_pct=-3.53),
            "obp": TargetResult(rmse=0.040, baseline_rmse=0.042, delta=-0.002, delta_pct=-4.76),
        }
        exp_id = repo.save(_make_experiment(target_results=target_results))

        result = repo.get(exp_id)
        assert result is not None
        assert len(result.target_results) == 2
        assert result.target_results["slg"].rmse == 0.082
        assert result.target_results["obp"].rmse == 0.040

    # --- find_by_target ---

    def test_find_by_target_returns_matching_sorted_by_delta_pct(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        # Experiment with slg delta_pct -5.0 (better)
        repo.save(
            _make_experiment(
                target_results={
                    "slg": TargetResult(rmse=0.080, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.0),
                },
            )
        )
        # Experiment with slg delta_pct -2.0 (worse)
        repo.save(
            _make_experiment(
                target_results={
                    "slg": TargetResult(rmse=0.083, baseline_rmse=0.085, delta=-0.002, delta_pct=-2.0),
                },
            )
        )
        # Experiment with only obp, no slg
        repo.save(
            _make_experiment(
                target_results={
                    "obp": TargetResult(rmse=0.040, baseline_rmse=0.042, delta=-0.002, delta_pct=-4.76),
                },
            )
        )

        results = repo.find_by_target("slg")
        assert len(results) == 2
        # Sorted by target delta_pct ascending (most negative first = best)
        assert results[0].target_results["slg"].delta_pct == -5.0
        assert results[1].target_results["slg"].delta_pct == -2.0

    def test_find_by_target_no_match(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment())  # has slg target
        assert repo.find_by_target("era") == []

    # --- find_by_feature ---

    def test_find_by_feature_in_added(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(feature_diff={"added": ["barrel_rate", "exit_velo"], "removed": []}))
        repo.save(_make_experiment(feature_diff={"added": ["sprint_speed"], "removed": []}))

        results = repo.find_by_feature("barrel_rate")
        assert len(results) == 1
        assert "barrel_rate" in results[0].feature_diff["added"]

    def test_find_by_feature_in_removed(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(feature_diff={"added": [], "removed": ["barrel_rate"]}))
        repo.save(_make_experiment(feature_diff={"added": ["sprint_speed"], "removed": []}))

        results = repo.find_by_feature("barrel_rate")
        assert len(results) == 1
        assert "barrel_rate" in results[0].feature_diff["removed"]

    def test_find_by_feature_finds_in_both_added_and_removed(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(
            _make_experiment(
                timestamp="2026-03-01T10:00:00",
                feature_diff={"added": ["barrel_rate"], "removed": []},
            )
        )
        repo.save(
            _make_experiment(
                timestamp="2026-03-01T11:00:00",
                feature_diff={"added": [], "removed": ["barrel_rate"]},
            )
        )
        repo.save(
            _make_experiment(
                timestamp="2026-03-01T12:00:00",
                feature_diff={"added": ["sprint_speed"], "removed": []},
            )
        )

        results = repo.find_by_feature("barrel_rate")
        assert len(results) == 2
        # Sorted by timestamp DESC
        assert results[0].timestamp == "2026-03-01T11:00:00"
        assert results[1].timestamp == "2026-03-01T10:00:00"

    def test_find_by_feature_no_match(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment())
        assert repo.find_by_feature("nonexistent_feature") == []

    # --- find_by_tag ---

    def test_find_by_tag(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(tags=["feature", "batter"]))
        repo.save(_make_experiment(tags=["hyperparameter"]))

        results = repo.find_by_tag("feature")
        assert len(results) == 1
        assert "feature" in results[0].tags

    def test_find_by_tag_no_match(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(tags=["feature"]))
        assert repo.find_by_tag("nonexistent") == []

    # --- find_by_model ---

    def test_find_by_model(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(model="model-a"))
        repo.save(_make_experiment(model="model-b"))
        repo.save(_make_experiment(model="model-a"))

        results = repo.find_by_model("model-a")
        assert len(results) == 2
        assert all(r.model == "model-a" for r in results)

    def test_find_by_model_no_match(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(model="model-a"))
        assert repo.find_by_model("nonexistent") == []

    # --- find_best ---

    def test_find_best_returns_experiment_with_lowest_delta_pct(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(
            _make_experiment(
                target_results={
                    "slg": TargetResult(rmse=0.080, baseline_rmse=0.085, delta=-0.005, delta_pct=-5.0),
                },
            )
        )
        repo.save(
            _make_experiment(
                target_results={
                    "slg": TargetResult(rmse=0.083, baseline_rmse=0.085, delta=-0.002, delta_pct=-2.0),
                },
            )
        )

        best = repo.find_best("slg")
        assert best is not None
        assert best.target_results["slg"].delta_pct == -5.0

    def test_find_best_returns_none_when_no_match(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment())  # has slg target
        assert repo.find_best("era") is None

    # --- find_recent ---

    def test_find_recent_respects_limit(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment(timestamp="2026-03-01T10:00:00"))
        repo.save(_make_experiment(timestamp="2026-03-01T12:00:00"))
        repo.save(_make_experiment(timestamp="2026-03-01T11:00:00"))

        results = repo.find_recent(2)
        assert len(results) == 2
        assert results[0].timestamp == "2026-03-01T12:00:00"
        assert results[1].timestamp == "2026-03-01T11:00:00"

    def test_find_recent_returns_all_when_limit_exceeds_count(self, conn: sqlite3.Connection) -> None:
        repo = SqliteExperimentRepo(conn)
        repo.save(_make_experiment())

        results = repo.find_recent(10)
        assert len(results) == 1
