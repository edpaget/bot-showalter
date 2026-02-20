from typing import Any

import pytest

from fantasy_baseball_manager.domain.evaluation import StatMetrics, SystemMetrics
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.models.ablation import (
    PlayerTypeConfig,
    evaluate_projections,
    multi_holdout_importance,
    run_ablation,
    single_holdout_importance,
)
from fantasy_baseball_manager.models.protocols import AblationResult, ModelConfig


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int | None]] = []

    def evaluate(
        self,
        system: str,
        version: str,
        season: int,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        top: int | None = None,
    ) -> SystemMetrics:
        self.calls.append((system, version, season, top))
        return SystemMetrics(
            system=system,
            version=version,
            source_type="test",
            metrics={"avg": StatMetrics(rmse=0.01, mae=0.008, correlation=0.9, r_squared=0.81, n=100)},
        )


class TestEvaluateProjections:
    def test_delegates_to_evaluator(self) -> None:
        evaluator = FakeEvaluator()
        config = ModelConfig(seasons=[2023], version="v1")
        result = evaluate_projections(evaluator, "test-model", config)
        assert isinstance(result, SystemMetrics)
        assert evaluator.calls == [("test-model", "v1", 2023, None)]

    def test_defaults_version_to_latest(self) -> None:
        evaluator = FakeEvaluator()
        config = ModelConfig(seasons=[2023])
        evaluate_projections(evaluator, "test-model", config)
        assert evaluator.calls[0][1] == "latest"

    def test_passes_top(self) -> None:
        evaluator = FakeEvaluator()
        config = ModelConfig(seasons=[2023], top=50)
        evaluate_projections(evaluator, "test-model", config)
        assert evaluator.calls[0][3] == 50

    def test_raises_without_evaluator(self) -> None:
        config = ModelConfig(seasons=[2023])
        with pytest.raises(TypeError):
            evaluate_projections(None, "test-model", config)


# ---------- Fake assembler for ablation tests ----------


class AblationFakeAssembler:
    """Assembler that stores rows by season and routes by feature set name."""

    def __init__(
        self,
        batter_rows_by_season: dict[int, list[dict[str, Any]]],
        pitcher_rows_by_season: dict[int, list[dict[str, Any]]],
    ) -> None:
        self._batter = batter_rows_by_season
        self._pitcher = pitcher_rows_by_season
        self._next_id = 1

    def _source(self, name: str) -> dict[int, list[dict[str, Any]]]:
        if "pitcher" in name:
            return self._pitcher
        return self._batter

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.get_or_materialize(feature_set)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        source = self._source(feature_set.name)
        rows: list[dict[str, Any]] = []
        for s in feature_set.seasons:
            rows.extend(source.get(s, []))
        handle = DatasetHandle(
            dataset_id=self._next_id,
            feature_set_id=self._next_id,
            table_name=f"ds_{feature_set.name}",
            row_count=len(rows),
            seasons=feature_set.seasons,
        )
        self._next_id += 1
        return handle

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits:
        source = self._source(handle.table_name)
        train_list = list(train)
        holdout_list = holdout or []
        train_rows: list[dict[str, Any]] = []
        for s in train_list:
            train_rows.extend(source.get(s, []))
        holdout_rows: list[dict[str, Any]] = []
        for s in holdout_list:
            holdout_rows.extend(source.get(s, []))
        train_handle = DatasetHandle(
            dataset_id=self._next_id,
            feature_set_id=handle.feature_set_id,
            table_name=f"{handle.table_name}_train",
            row_count=len(train_rows),
            seasons=tuple(train_list),
        )
        self._next_id += 1
        holdout_handle: DatasetHandle | None = None
        if holdout_rows:
            holdout_handle = DatasetHandle(
                dataset_id=self._next_id,
                feature_set_id=handle.feature_set_id,
                table_name=f"{handle.table_name}_holdout",
                row_count=len(holdout_rows),
                seasons=tuple(holdout_list),
            )
            self._next_id += 1
        return DatasetSplits(train=train_handle, validation=None, holdout=holdout_handle)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        source = self._source(handle.table_name)
        rows: list[dict[str, Any]] = []
        for s in handle.seasons:
            rows.extend(source.get(s, []))
        return rows


def _make_batter_row(pid: int, season: int) -> dict[str, Any]:
    return {
        "player_id": pid,
        "season": season,
        "feat_a": float(pid) * 0.1 + season * 0.01,
        "feat_b": float(pid) * 0.2 + season * 0.02,
        "target_avg": 0.270 + pid * 0.001,
        "target_obp": 0.340 + pid * 0.001,
        "target_slg": 0.440 + pid * 0.001,
        "target_woba": 0.330 + pid * 0.001,
        "target_h": 145 + pid,
        "target_hr": 24 + pid,
        "target_ab": 500 + pid,
        "target_so": 110 + pid,
        "target_sf": 4,
    }


def _make_pitcher_row(pid: int, season: int) -> dict[str, Any]:
    return {
        "player_id": pid,
        "season": season,
        "feat_c": float(pid) * 0.15 + season * 0.01,
        "feat_d": float(pid) * 0.25 + season * 0.02,
        "target_era": 3.60 + pid * 0.01,
        "target_fip": 3.50 + pid * 0.01,
        "target_k_per_9": 9.5 + pid * 0.01,
        "target_bb_per_9": 2.8 + pid * 0.01,
        "target_whip": 1.18 + pid * 0.001,
        "target_h": 155 + pid,
        "target_hr": 18 + pid,
        "target_ip": 175.0 + pid,
        "target_so": 190 + pid,
    }


def _build_assembler(n_batters: int, n_pitchers: int, seasons: list[int]) -> AblationFakeAssembler:
    batter_rows = {s: [_make_batter_row(i, s) for i in range(1, n_batters + 1)] for s in seasons}
    pitcher_rows = {s: [_make_pitcher_row(i, s) for i in range(100, 100 + n_pitchers)] for s in seasons}
    return AblationFakeAssembler(batter_rows, pitcher_rows)


BAT_COLS = ["feat_a", "feat_b"]
BAT_TARGETS = ["avg", "obp", "slg", "woba"]
PITCH_COLS = ["feat_c", "feat_d"]
PITCH_TARGETS = ["era", "fip", "k_per_9", "bb_per_9", "whip"]

BAT_FS = FeatureSet(name="test_batter_train", features=(), seasons=(2022, 2023))
PITCH_FS = FeatureSet(name="test_pitcher_train", features=(), seasons=(2022, 2023))


@pytest.mark.slow
class TestSingleHoldoutImportance:
    def test_returns_grouped_importance_result(self) -> None:
        assembler = _build_assembler(30, 0, [2022, 2023])
        fs = FeatureSet(name="test_batter_train", features=(), seasons=(2022, 2023))
        result = single_holdout_importance(
            assembler, fs, BAT_COLS, BAT_TARGETS, {}, [2022], [2023], n_repeats=5, correlation_threshold=0.70
        )
        assert result is not None
        assert set(result.feature_importance.keys()) == {"feat_a", "feat_b"}

    def test_returns_none_without_holdout(self) -> None:
        assembler = _build_assembler(30, 0, [2022])
        fs = FeatureSet(name="test_batter_train", features=(), seasons=(2022,))
        # Split with no holdout season — assembler returns holdout=None when no holdout rows
        result = single_holdout_importance(
            assembler, fs, BAT_COLS, BAT_TARGETS, {}, [2022], [], n_repeats=5, correlation_threshold=0.70
        )
        # When holdout is empty, single_holdout returns None
        assert result is None


@pytest.mark.slow
class TestMultiHoldoutImportance:
    def test_returns_grouped_importance_result(self) -> None:
        assembler = _build_assembler(30, 0, [2021, 2022, 2023])
        fs = FeatureSet(name="test_batter_train", features=(), seasons=(2021, 2022, 2023))
        result = multi_holdout_importance(
            assembler, fs, BAT_COLS, BAT_TARGETS, {}, n_repeats=5, correlation_threshold=0.70
        )
        assert set(result.feature_importance.keys()) == {"feat_a", "feat_b"}


@pytest.mark.slow
class TestRunAblation:
    @pytest.fixture(scope="class")
    def ablation_result(self) -> AblationResult:
        assembler = _build_assembler(30, 30, [2022, 2023])
        bat_config = PlayerTypeConfig(
            player_type="batter",
            train_fs=FeatureSet(name="test_batter_train", features=(), seasons=(2022, 2023)),
            columns=BAT_COLS,
            targets=BAT_TARGETS,
            params={},
        )
        pitch_config = PlayerTypeConfig(
            player_type="pitcher",
            train_fs=FeatureSet(name="test_pitcher_train", features=(), seasons=(2022, 2023)),
            columns=PITCH_COLS,
            targets=PITCH_TARGETS,
            params={},
        )
        config = ModelConfig(
            seasons=[2022, 2023],
            model_params={"n_repeats": 5},
        )
        return run_ablation(assembler, "test-model", config, [bat_config, pitch_config])

    def test_returns_ablation_result_with_prefixed_keys(self, ablation_result: AblationResult) -> None:
        assert isinstance(ablation_result, AblationResult)
        assert ablation_result.model_name == "test-model"
        batter_keys = [k for k in ablation_result.feature_impacts if k.startswith("batter:")]
        pitcher_keys = [k for k in ablation_result.feature_impacts if k.startswith("pitcher:")]
        assert len(batter_keys) > 0
        assert len(pitcher_keys) > 0

    def test_standard_errors_non_negative(self, ablation_result: AblationResult) -> None:
        for v in ablation_result.feature_standard_errors.values():
            assert v >= 0.0

    def test_no_validation_by_default(self, ablation_result: AblationResult) -> None:
        assert ablation_result.validation_results == {}
