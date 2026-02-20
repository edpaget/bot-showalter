import math
from pathlib import Path
from typing import Any

import pytest
from fantasy_baseball_manager.domain.evaluation import SystemMetrics
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
from fantasy_baseball_manager.models.protocols import (
    Ablatable,
    AblationResult,
    Evaluable,
    FeatureIntrospectable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    PredictResult,
    Preparable,
    PrepareResult,
    Trainable,
    TrainResult,
    Tunable,
    TuneResult,
)
from fantasy_baseball_manager.models.statcast_gbm.features import (
    batter_preseason_feature_columns,
    live_batter_curated_columns,
    live_pitcher_curated_columns,
    pitcher_preseason_feature_columns,
)
from fantasy_baseball_manager.models import ablation as ablation_mod
from fantasy_baseball_manager.models.statcast_gbm import model as statcast_gbm_model_mod
from fantasy_baseball_manager.models.statcast_gbm.model import StatcastGBMModel, StatcastGBMPreseasonModel
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS

pytestmark = pytest.mark.slow

_FEATURE_COLUMNS = live_batter_curated_columns()
_PITCHER_FEATURE_COLUMNS = live_pitcher_curated_columns()
_PRESEASON_FEATURE_COLUMNS = batter_preseason_feature_columns()
_PRESEASON_PITCHER_FEATURE_COLUMNS = pitcher_preseason_feature_columns()


def _make_row(player_id: str, season: int) -> dict[str, Any]:
    row: dict[str, Any] = {"player_id": player_id, "season": season}
    for col in _FEATURE_COLUMNS:
        row[col] = 1.0
    # Target columns for training
    row["target_avg"] = 0.275
    row["target_obp"] = 0.350
    row["target_slg"] = 0.450
    row["target_woba"] = 0.340
    row["target_h"] = 150
    row["target_hr"] = 25
    row["target_ab"] = 500
    row["target_so"] = 100
    row["target_sf"] = 5
    # Metadata columns
    row["pa"] = 500
    row["war"] = 2.0
    return row


def _make_rows(n: int, season: int) -> list[dict[str, Any]]:
    return [_make_row(f"player_{i}", season) for i in range(n)]


def _make_pitcher_row(player_id: str, season: int) -> dict[str, Any]:
    row: dict[str, Any] = {"player_id": player_id, "season": season}
    for col in _PITCHER_FEATURE_COLUMNS:
        row[col] = 1.0
    # Target columns for training
    row["target_era"] = 3.50
    row["target_fip"] = 3.40
    row["target_k_per_9"] = 9.0
    row["target_bb_per_9"] = 3.0
    row["target_whip"] = 1.20
    row["target_h"] = 150
    row["target_hr"] = 20
    row["target_ip"] = 180.0
    row["target_so"] = 160
    # Metadata columns
    row["ip"] = 180.0
    row["war"] = 2.0
    return row


def _make_pitcher_rows(n: int, season: int) -> list[dict[str, Any]]:
    return [_make_pitcher_row(f"pitcher_{i}", season) for i in range(n)]


class FakeAssembler:
    def __init__(
        self,
        rows_by_season: dict[int, list[dict[str, Any]]] | None = None,
        pitcher_rows_by_season: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._next_id = 1
        self._rows_by_season = rows_by_season or {}
        self._pitcher_rows_by_season = pitcher_rows_by_season or {}
        self._handles: dict[int, FeatureSet] = {}

    def _select_rows(self, feature_set_name: str) -> dict[int, list[dict[str, Any]]]:
        if "pitching" in feature_set_name:
            return self._pitcher_rows_by_season
        return self._rows_by_season

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.get_or_materialize(feature_set)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        source = self._select_rows(feature_set.name)
        all_rows = []
        for s in feature_set.seasons:
            all_rows.extend(source.get(s, []))
        handle = DatasetHandle(
            dataset_id=self._next_id,
            feature_set_id=self._next_id,
            table_name=f"ds_{feature_set.name}",
            row_count=len(all_rows),
            seasons=feature_set.seasons,
        )
        self._handles[handle.dataset_id] = feature_set
        self._next_id += 1
        return handle

    def split(
        self,
        handle: DatasetHandle,
        train: range | list[int],
        validation: list[int] | None = None,
        holdout: list[int] | None = None,
    ) -> DatasetSplits:
        source = self._select_rows(handle.table_name)
        train_seasons = list(train)
        holdout_seasons = holdout or []
        train_rows = []
        for s in train_seasons:
            train_rows.extend(source.get(s, []))
        holdout_rows = []
        for s in holdout_seasons:
            holdout_rows.extend(source.get(s, []))
        train_handle = DatasetHandle(
            dataset_id=self._next_id,
            feature_set_id=handle.feature_set_id,
            table_name=f"{handle.table_name}_train",
            row_count=len(train_rows),
            seasons=tuple(train_seasons),
        )
        self._next_id += 1
        holdout_handle = DatasetHandle(
            dataset_id=self._next_id,
            feature_set_id=handle.feature_set_id,
            table_name=f"{handle.table_name}_holdout",
            row_count=len(holdout_rows),
            seasons=tuple(holdout_seasons),
        )
        self._next_id += 1
        return DatasetSplits(train=train_handle, validation=None, holdout=holdout_handle)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        source = self._select_rows(handle.table_name)
        rows: list[dict[str, Any]] = []
        for s in handle.seasons:
            rows.extend(source.get(s, []))
        return rows


class FakeEvaluator:
    def __init__(self) -> None:
        self.called_with: dict[str, Any] = {}

    def evaluate(
        self,
        system: str,
        version: str,
        season: int,
        stats: list[str] | None = None,
        actuals_source: str = "fangraphs",
        top: int | None = None,
    ) -> SystemMetrics:
        self.called_with = {
            "system": system,
            "version": version,
            "season": season,
            "stats": stats,
            "actuals_source": actuals_source,
        }
        return SystemMetrics(
            system=system,
            version=version,
            source_type="first_party",
            metrics={},
        )


_NULL_ASSEMBLER = FakeAssembler()
_NULL_EVALUATOR = FakeEvaluator()


class TestStatcastGBMProtocol:
    def test_is_model(self) -> None:
        assert isinstance(StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Preparable)

    def test_is_trainable(self) -> None:
        assert isinstance(StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Trainable)

    def test_is_evaluable(self) -> None:
        assert isinstance(StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Evaluable)

    def test_is_predictable(self) -> None:
        assert isinstance(StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Predictable)

    def test_is_ablatable(self) -> None:
        assert isinstance(StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Ablatable)

    def test_is_feature_introspectable(self) -> None:
        assert isinstance(StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), FeatureIntrospectable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), FineTunable)

    def test_name(self) -> None:
        assert StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR).name == "statcast-gbm"

    def test_artifact_type(self) -> None:
        assert StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR).artifact_type == "file"

    def test_supported_operations(self) -> None:
        ops = StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR).supported_operations
        assert ops == frozenset({"prepare", "train", "evaluate", "predict", "ablate", "tune"})

    def test_declared_features_not_empty(self) -> None:
        features = StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR).declared_features
        assert len(features) > 0


class TestStatcastGBMPreseasonProtocol:
    def test_is_model(self) -> None:
        assert isinstance(StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Preparable)

    def test_is_trainable(self) -> None:
        assert isinstance(StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Trainable)

    def test_is_evaluable(self) -> None:
        assert isinstance(StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Evaluable)

    def test_is_predictable(self) -> None:
        assert isinstance(StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Predictable)

    def test_is_ablatable(self) -> None:
        assert isinstance(StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Ablatable)

    def test_is_feature_introspectable(self) -> None:
        assert isinstance(
            StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), FeatureIntrospectable
        )

    def test_name(self) -> None:
        assert (
            StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR).name
            == "statcast-gbm-preseason"
        )

    def test_artifact_type(self) -> None:
        assert StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR).artifact_type == "file"

    def test_supported_operations(self) -> None:
        ops = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR).supported_operations
        assert ops == frozenset({"prepare", "train", "evaluate", "predict", "ablate", "tune", "sweep"})

    def test_declared_features_not_empty(self) -> None:
        features = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR).declared_features
        assert len(features) > 0


class TestSampleWeightColumns:
    def test_preseason_batter_weight_column(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._batter_sample_weight_column == "pa_1"

    def test_preseason_pitcher_weight_column(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._pitcher_sample_weight_column == "ip_1"

    def test_live_model_weight_columns_none(self) -> None:
        model = StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._batter_sample_weight_column is None
        assert model._pitcher_sample_weight_column is None


class TestStatcastGBMPrepare:
    def test_prepare_returns_result(self) -> None:
        assembler = FakeAssembler()
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2023])
        result = model.prepare(config)
        assert isinstance(result, PrepareResult)
        assert result.model_name == "statcast-gbm"


class TestStatcastGBMTrain:
    def test_train_returns_metrics(self, tmp_path: Path) -> None:
        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        result = model.train(config)
        assert isinstance(result, TrainResult)
        assert result.model_name == "statcast-gbm"
        for target in BATTER_TARGETS:
            assert f"batter_rmse_{target}" in result.metrics
        for target in PITCHER_TARGETS:
            assert f"pitcher_rmse_{target}" in result.metrics

    def test_train_saves_artifact(self, tmp_path: Path) -> None:
        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        model.train(config)
        batter_path = tmp_path / "statcast-gbm" / "latest" / "batter_models.joblib"
        pitcher_path = tmp_path / "statcast-gbm" / "latest" / "pitcher_models.joblib"
        assert batter_path.exists()
        assert pitcher_path.exists()


class TestStatcastGBMPredict:
    def test_predict_returns_predictions(self, tmp_path: Path) -> None:
        # First train to create artifact
        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        model.train(config)

        # Now predict
        result = model.predict(config)
        assert isinstance(result, PredictResult)
        assert result.model_name == "statcast-gbm"
        assert len(result.predictions) > 0

    def test_predict_prediction_shape(self, tmp_path: Path) -> None:
        rows_by_season = {
            2022: _make_rows(5, 2022),
            2023: _make_rows(5, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(5, 2022),
            2023: _make_pitcher_rows(5, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        model.train(config)
        result = model.predict(config)
        batter_preds = [p for p in result.predictions if p["player_type"] == "batter"]
        pitcher_preds = [p for p in result.predictions if p["player_type"] == "pitcher"]
        for pred in batter_preds:
            assert "player_id" in pred
            assert "season" in pred
            for target in BATTER_TARGETS:
                assert target in pred
        for pred in pitcher_preds:
            assert "player_id" in pred
            assert "season" in pred
            for target in PITCHER_TARGETS:
                assert target in pred

    def test_predict_has_both_player_types(self, tmp_path: Path) -> None:
        rows_by_season = {
            2022: _make_rows(5, 2022),
            2023: _make_rows(5, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(5, 2022),
            2023: _make_pitcher_rows(5, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        model.train(config)
        result = model.predict(config)
        player_types = {p["player_type"] for p in result.predictions}
        assert "batter" in player_types
        assert "pitcher" in player_types


class TestStatcastGBMEvaluate:
    def test_evaluate_delegates_to_evaluator(self) -> None:
        evaluator = FakeEvaluator()
        model = StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=evaluator)
        config = ModelConfig(seasons=[2023], version="v1")
        result = model.evaluate(config)
        assert isinstance(result, SystemMetrics)
        assert result.system == "statcast-gbm"
        assert evaluator.called_with["system"] == "statcast-gbm"
        assert evaluator.called_with["version"] == "v1"
        assert evaluator.called_with["season"] == 2023


@pytest.fixture(scope="class")
def ablation_result() -> AblationResult:
    rows_by_season = {
        2022: _make_rows(10, 2022),
        2023: _make_rows(10, 2023),
    }
    pitcher_rows_by_season = {
        2022: _make_pitcher_rows(10, 2022),
        2023: _make_pitcher_rows(10, 2023),
    }
    assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
    model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
    config = ModelConfig(seasons=[2022, 2023])
    return model.ablate(config)


class TestStatcastGBMAblate:
    def test_ablate_returns_result(self, ablation_result: AblationResult) -> None:
        assert isinstance(ablation_result, AblationResult)
        assert ablation_result.model_name == "statcast-gbm"
        assert isinstance(ablation_result.feature_impacts, dict)

    def test_ablate_returns_nonempty_impacts(self, ablation_result: AblationResult) -> None:
        assert len(ablation_result.feature_impacts) > 0

    def test_ablate_impacts_include_batter_features(self, ablation_result: AblationResult) -> None:
        batter_keys = [k for k in ablation_result.feature_impacts if k.startswith("batter:")]
        assert len(batter_keys) > 0

    def test_ablate_impacts_include_pitcher_features(self, ablation_result: AblationResult) -> None:
        pitcher_keys = [k for k in ablation_result.feature_impacts if k.startswith("pitcher:")]
        assert len(pitcher_keys) > 0

    def test_ablate_returns_standard_errors(self, ablation_result: AblationResult) -> None:
        assert len(ablation_result.feature_standard_errors) > 0

    def test_ablate_standard_errors_match_impact_keys(self, ablation_result: AblationResult) -> None:
        assert set(ablation_result.feature_standard_errors.keys()) == set(ablation_result.feature_impacts.keys())

    def test_ablate_standard_errors_non_negative(self, ablation_result: AblationResult) -> None:
        for se in ablation_result.feature_standard_errors.values():
            assert se >= 0

    def test_ablate_returns_group_impacts(self, ablation_result: AblationResult) -> None:
        assert isinstance(ablation_result.group_impacts, dict)

    def test_ablate_group_keys_prefixed(self, ablation_result: AblationResult) -> None:
        for key in ablation_result.group_impacts:
            assert key.startswith("batter:") or key.startswith("pitcher:")

    def test_ablate_group_members_are_prefixed(self, ablation_result: AblationResult) -> None:
        for members in ablation_result.group_members.values():
            for m in members:
                assert m.startswith("batter:") or m.startswith("pitcher:")

    def test_ablate_group_standard_errors_non_negative(self, ablation_result: AblationResult) -> None:
        for se in ablation_result.group_standard_errors.values():
            assert se >= 0


@pytest.fixture(scope="class")
def preseason_ablation_result() -> AblationResult:
    rows_by_season = {
        2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
        2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
    }
    pitcher_rows_by_season = {
        2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
        2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
    }
    assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
    model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
    config = ModelConfig(seasons=[2022, 2023])
    return model.ablate(config)


class TestStatcastGBMPreseasonAblate:
    def test_ablate_preseason_returns_result(self, preseason_ablation_result: AblationResult) -> None:
        assert isinstance(preseason_ablation_result, AblationResult)
        assert preseason_ablation_result.model_name == "statcast-gbm-preseason"

    def test_ablate_preseason_returns_nonempty_impacts(self, preseason_ablation_result: AblationResult) -> None:
        assert len(preseason_ablation_result.feature_impacts) > 0

    def test_ablate_preseason_includes_batter_features(self, preseason_ablation_result: AblationResult) -> None:
        batter_keys = [k for k in preseason_ablation_result.feature_impacts if k.startswith("batter:")]
        assert len(batter_keys) > 0

    def test_ablate_preseason_includes_pitcher_features(self, preseason_ablation_result: AblationResult) -> None:
        pitcher_keys = [k for k in preseason_ablation_result.feature_impacts if k.startswith("pitcher:")]
        assert len(pitcher_keys) > 0


class TestStatcastGBMTrainWithMissingTargets:
    def test_train_handles_missing_target_columns(self, tmp_path: Path) -> None:
        # Build batter rows where some are missing target_slg (breaks iso but not avg)
        good_rows = _make_rows(8, 2022)
        bad_rows = _make_rows(2, 2022)
        for row in bad_rows:
            del row["target_slg"]
        rows_2022 = good_rows + bad_rows

        good_rows_2023 = _make_rows(8, 2023)
        bad_rows_2023 = _make_rows(2, 2023)
        for row in bad_rows_2023:
            del row["target_slg"]
        rows_2023 = good_rows_2023 + bad_rows_2023

        rows_by_season = {2022: rows_2022, 2023: rows_2023}
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        result = model.train(config)
        assert isinstance(result, TrainResult)
        for target in BATTER_TARGETS:
            assert f"batter_rmse_{target}" in result.metrics


class TestStatcastGBMSeasonValidation:
    def test_train_empty_seasons(self) -> None:
        model = StatcastGBMModel(assembler=FakeAssembler(), evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[])
        with pytest.raises(ValueError, match="train requires at least 2 seasons"):
            model.train(config)

    def test_train_single_season(self) -> None:
        model = StatcastGBMModel(assembler=FakeAssembler(), evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2023])
        with pytest.raises(ValueError, match="train requires at least 2 seasons"):
            model.train(config)

    def test_ablate_empty_seasons(self) -> None:
        model = StatcastGBMModel(assembler=FakeAssembler(), evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[])
        with pytest.raises(ValueError, match="ablate requires at least 2 seasons"):
            model.ablate(config)

    def test_ablate_single_season(self) -> None:
        model = StatcastGBMModel(assembler=FakeAssembler(), evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2023])
        with pytest.raises(ValueError, match="ablate requires at least 2 seasons"):
            model.ablate(config)


def _make_preseason_row(player_id: str, season: int) -> dict[str, Any]:
    row: dict[str, Any] = {"player_id": player_id, "season": season}
    for col in _PRESEASON_FEATURE_COLUMNS:
        row[col] = 1.0
    row["target_avg"] = 0.275
    row["target_obp"] = 0.350
    row["target_slg"] = 0.450
    row["target_woba"] = 0.340
    row["target_h"] = 150
    row["target_hr"] = 25
    row["target_ab"] = 500
    row["target_so"] = 100
    row["target_sf"] = 5
    # Metadata columns
    row["pa"] = 500
    row["war"] = 2.0
    return row


def _make_preseason_pitcher_row(player_id: str, season: int) -> dict[str, Any]:
    row: dict[str, Any] = {"player_id": player_id, "season": season}
    for col in _PRESEASON_PITCHER_FEATURE_COLUMNS:
        row[col] = 1.0
    row["target_era"] = 3.50
    row["target_fip"] = 3.40
    row["target_k_per_9"] = 9.0
    row["target_bb_per_9"] = 3.0
    row["target_whip"] = 1.20
    row["target_h"] = 150
    row["target_hr"] = 20
    row["target_ip"] = 180.0
    row["target_so"] = 160
    # Metadata columns
    row["ip"] = 180.0
    row["war"] = 2.0
    return row


class TestStatcastGBMPreseasonTrain:
    def test_train_preseason_returns_metrics(self, tmp_path: Path) -> None:
        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        result = model.train(config)
        assert isinstance(result, TrainResult)
        for target in BATTER_TARGETS:
            assert f"batter_rmse_{target}" in result.metrics

    def test_train_preseason_saves_artifacts(self, tmp_path: Path) -> None:
        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        model.train(config)
        batter_path = tmp_path / "statcast-gbm-preseason" / "latest" / "batter_models.joblib"
        pitcher_path = tmp_path / "statcast-gbm-preseason" / "latest" / "pitcher_models.joblib"
        assert batter_path.exists()
        assert pitcher_path.exists()


class TestStatcastGBMPreseasonPredict:
    def test_predict_preseason(self, tmp_path: Path) -> None:
        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        model.train(config)
        result = model.predict(config)
        assert isinstance(result, PredictResult)
        assert len(result.predictions) > 0


class TestStatcastGBMPreseasonTune:
    @pytest.fixture(scope="class")
    def tune_result(self) -> TuneResult:
        rows_by_season = {
            2021: [_make_preseason_row(f"p_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows_by_season = {
            2021: [_make_preseason_pitcher_row(f"pit_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            model_params={"param_grid": {"max_iter": [50, 100]}},
        )
        return model.tune(config)

    def test_tune_returns_tune_result(self, tune_result: TuneResult) -> None:
        assert isinstance(tune_result, TuneResult)
        assert tune_result.model_name == "statcast-gbm-preseason"

    def test_batter_params_nonempty(self, tune_result: TuneResult) -> None:
        assert isinstance(tune_result.batter_params, dict)
        assert len(tune_result.batter_params) > 0

    def test_pitcher_params_nonempty(self, tune_result: TuneResult) -> None:
        assert isinstance(tune_result.pitcher_params, dict)
        assert len(tune_result.pitcher_params) > 0

    def test_batter_cv_rmse_has_entries_for_all_targets(self, tune_result: TuneResult) -> None:
        for target in BATTER_TARGETS:
            assert target in tune_result.batter_cv_rmse

    def test_pitcher_cv_rmse_has_entries_for_all_targets(self, tune_result: TuneResult) -> None:
        for target in PITCHER_TARGETS:
            assert target in tune_result.pitcher_cv_rmse

    def test_tune_raises_with_fewer_than_3_seasons(self) -> None:
        assembler = FakeAssembler()
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2022, 2023])
        with pytest.raises(ValueError, match="at least 3 seasons"):
            model.tune(config)

    def test_tune_raises_with_single_season(self) -> None:
        assembler = FakeAssembler()
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2023])
        with pytest.raises(ValueError, match="at least 3 seasons"):
            model.tune(config)

    def test_is_tunable_protocol(self) -> None:
        assert isinstance(StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR), Tunable)

    def test_supported_operations_includes_tune(self) -> None:
        ops = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR).supported_operations
        assert "tune" in ops


class TestTrainSampleWeights:
    def test_train_preseason_passes_sample_weights(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_weights: list[list[float] | None] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_weights.append(kwargs.get("sample_weights"))
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2022, 2023], artifacts_dir=str(tmp_path))
        model.train(config)
        # Both batter and pitcher should have weights
        assert len(captured_weights) == 2
        assert captured_weights[0] is not None
        assert isinstance(captured_weights[0], list)
        assert all(isinstance(w, float) for w in captured_weights[0])
        assert captured_weights[1] is not None
        assert isinstance(captured_weights[1], list)
        assert all(isinstance(w, float) for w in captured_weights[1])

    def test_train_live_no_sample_weights(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_weights: list[list[float] | None] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_weights.append(kwargs.get("sample_weights"))
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2022, 2023], artifacts_dir=str(tmp_path))
        model.train(config)
        assert len(captured_weights) == 2
        assert captured_weights[0] is None
        assert captured_weights[1] is None


class TestTuneSampleWeights:
    def test_tune_preseason_uses_sample_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_folds: list[list[Any]] = []
        original_grid = statcast_gbm_model_mod.grid_search_cv

        def spy_grid(folds: list[Any], *args: Any, **kwargs: Any) -> Any:
            captured_folds.append(folds)
            return original_grid(folds, *args, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "grid_search_cv", spy_grid)

        rows_by_season = {
            2021: [_make_preseason_row(f"p_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2021: [_make_preseason_pitcher_row(f"pit_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            model_params={"param_grid": {"max_iter": [100]}},
        )
        model.tune(config)
        # grid_search_cv called twice (batter + pitcher)
        assert len(captured_folds) == 2
        # Each fold list should have folds with sample_weights set
        for fold_list in captured_folds:
            for fold in fold_list:
                assert fold.sample_weights is not None
                assert isinstance(fold.sample_weights, list)
                assert all(isinstance(w, float) for w in fold.sample_weights)


class TestStatcastGBMTrainPerTypeParams:
    def test_train_routes_per_type_params(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_params: list[dict[str, Any]] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_params.append(params)
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
            model_params={
                "batter": {"max_iter": 50, "min_samples_leaf": 10},
                "pitcher": {"max_iter": 80, "learning_rate": 0.05},
            },
        )
        result = model.train(config)
        assert isinstance(result, TrainResult)
        # First call is batter, second is pitcher
        assert len(captured_params) == 2
        assert captured_params[0] == {"max_iter": 50, "min_samples_leaf": 10}
        assert captured_params[1] == {"max_iter": 80, "learning_rate": 0.05}

    def test_train_falls_back_to_top_level_params(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_params: list[dict[str, Any]] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_params.append(params)
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
            model_params={"max_iter": 50},
        )
        result = model.train(config)
        assert isinstance(result, TrainResult)
        # Both batter and pitcher get the same top-level params
        assert len(captured_params) == 2
        assert captured_params[0] == {"max_iter": 50}
        assert captured_params[1] == {"max_iter": 50}


class TestStatcastGBMAblateNRepeats:
    def test_ablate_passes_n_repeats_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_n_repeats: list[int] = []
        original_cgpi = ablation_mod.compute_grouped_permutation_importance

        def spy_cgpi(*args: Any, **kwargs: Any) -> Any:
            captured_n_repeats.append(kwargs.get("n_repeats", 20))
            return original_cgpi(*args, **kwargs)

        monkeypatch.setattr(ablation_mod, "compute_grouped_permutation_importance", spy_cgpi)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            model_params={"n_repeats": 10},
        )
        model.ablate(config)
        assert len(captured_n_repeats) == 2
        assert captured_n_repeats[0] == 10
        assert captured_n_repeats[1] == 10

    def test_ablate_uses_default_n_repeats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_n_repeats: list[int] = []
        original_cgpi = ablation_mod.compute_grouped_permutation_importance

        def spy_cgpi(*args: Any, **kwargs: Any) -> Any:
            captured_n_repeats.append(kwargs.get("n_repeats", 20))
            return original_cgpi(*args, **kwargs)

        monkeypatch.setattr(ablation_mod, "compute_grouped_permutation_importance", spy_cgpi)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2022, 2023])
        model.ablate(config)
        assert len(captured_n_repeats) == 2
        assert captured_n_repeats[0] == 20
        assert captured_n_repeats[1] == 20


class TestStatcastGBMAblatePerTypeParams:
    def test_ablate_routes_per_type_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_params: list[dict[str, Any]] = []
        original_fit = ablation_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any]) -> Any:
            captured_params.append(params)
            return original_fit(X, y, params)

        monkeypatch.setattr(ablation_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            model_params={
                "batter": {"max_iter": 50, "min_samples_leaf": 10},
                "pitcher": {"max_iter": 80, "learning_rate": 0.05},
            },
        )
        result = model.ablate(config)
        assert isinstance(result, AblationResult)
        # First call is batter, second is pitcher
        assert len(captured_params) == 2
        assert captured_params[0] == {"max_iter": 50, "min_samples_leaf": 10}
        assert captured_params[1] == {"max_iter": 80, "learning_rate": 0.05}

    def test_ablate_falls_back_to_top_level_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_params: list[dict[str, Any]] = []
        original_fit = ablation_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any]) -> Any:
            captured_params.append(params)
            return original_fit(X, y, params)

        monkeypatch.setattr(ablation_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            model_params={"max_iter": 50},
        )
        result = model.ablate(config)
        assert isinstance(result, AblationResult)
        # Both batter and pitcher get the same top-level params
        assert len(captured_params) == 2
        assert captured_params[0] == {"max_iter": 50}
        assert captured_params[1] == {"max_iter": 50}


class TestStatcastGBMAblateCorrelationThreshold:
    def test_ablate_passes_correlation_threshold_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_thresholds: list[float] = []
        original_cgpi = ablation_mod.compute_grouped_permutation_importance

        def spy_cgpi(*args: Any, **kwargs: Any) -> Any:
            captured_thresholds.append(kwargs.get("correlation_threshold", 0.70))
            return original_cgpi(*args, **kwargs)

        monkeypatch.setattr(ablation_mod, "compute_grouped_permutation_importance", spy_cgpi)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            model_params={"correlation_threshold": 0.80},
        )
        model.ablate(config)
        assert len(captured_thresholds) == 2
        assert captured_thresholds[0] == 0.80
        assert captured_thresholds[1] == 0.80

    def test_ablate_uses_default_correlation_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_thresholds: list[float] = []
        original_cgpi = ablation_mod.compute_grouped_permutation_importance

        def spy_cgpi(*args: Any, **kwargs: Any) -> Any:
            captured_thresholds.append(kwargs.get("correlation_threshold", 0.70))
            return original_cgpi(*args, **kwargs)

        monkeypatch.setattr(ablation_mod, "compute_grouped_permutation_importance", spy_cgpi)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2022, 2023])
        model.ablate(config)
        assert len(captured_thresholds) == 2
        assert captured_thresholds[0] == 0.70
        assert captured_thresholds[1] == 0.70


class TestDefaultModeIsTrueTalent:
    def test_artifact_path_uses_model_name(self, tmp_path: Path) -> None:
        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        model.train(config)
        batter_path = tmp_path / "statcast-gbm" / "latest" / "batter_models.joblib"
        assert batter_path.exists()


class TestStatcastGBMAblateValidation:
    def test_ablate_default_has_no_validation(self, ablation_result: AblationResult) -> None:
        assert ablation_result.validation_results == {}

    def test_ablate_validate_true_calls_validate_pruning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []
        original_vp = ablation_mod.validate_pruning

        def spy_vp(*args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return original_vp(*args, **kwargs)

        # Make identify_prune_candidates always return something to trigger validation
        monkeypatch.setattr(ablation_mod, "identify_prune_candidates", lambda result: ["fake_col"])
        monkeypatch.setattr(ablation_mod, "validate_pruning", spy_vp)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            model_params={"validate": True},
        )
        result = model.ablate(config)
        assert isinstance(result, AblationResult)
        # validate_pruning called for both batter and pitcher
        assert len(calls) == 2

    def test_ablate_validate_passes_max_degradation_pct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_max_deg: list[float] = []
        original_vp = ablation_mod.validate_pruning

        def spy_vp(*args: Any, **kwargs: Any) -> Any:
            captured_max_deg.append(kwargs.get("max_degradation_pct", 5.0))
            return original_vp(*args, **kwargs)

        monkeypatch.setattr(ablation_mod, "identify_prune_candidates", lambda result: ["fake_col"])
        monkeypatch.setattr(ablation_mod, "validate_pruning", spy_vp)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            model_params={"validate": True, "max_degradation_pct": 3.0},
        )
        model.ablate(config)
        assert len(captured_max_deg) == 2
        assert captured_max_deg[0] == 3.0
        assert captured_max_deg[1] == 3.0

    def test_ablate_validate_skips_when_no_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vp_calls: list[Any] = []
        original_vp = ablation_mod.validate_pruning

        def spy_vp(*args: Any, **kwargs: Any) -> Any:
            vp_calls.append(True)
            return original_vp(*args, **kwargs)

        monkeypatch.setattr(ablation_mod, "identify_prune_candidates", lambda result: [])
        monkeypatch.setattr(ablation_mod, "validate_pruning", spy_vp)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            model_params={"validate": True},
        )
        result = model.ablate(config)
        assert isinstance(result, AblationResult)
        assert len(vp_calls) == 0
        assert result.validation_results == {}


@pytest.fixture(scope="class")
def multi_holdout_result() -> AblationResult:
    rows_by_season = {
        2021: _make_rows(10, 2021),
        2022: _make_rows(10, 2022),
        2023: _make_rows(10, 2023),
    }
    pitcher_rows_by_season = {
        2021: _make_pitcher_rows(10, 2021),
        2022: _make_pitcher_rows(10, 2022),
        2023: _make_pitcher_rows(10, 2023),
    }
    assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
    model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
    config = ModelConfig(
        seasons=[2021, 2022, 2023],
        model_params={"multi_holdout": True, "n_repeats": 5},
    )
    return model.ablate(config)


class TestStatcastGBMAblateMultiHoldout:
    def test_multi_holdout_returns_ablation_result(self, multi_holdout_result: AblationResult) -> None:
        assert isinstance(multi_holdout_result, AblationResult)
        assert multi_holdout_result.model_name == "statcast-gbm"

    def test_multi_holdout_has_batter_and_pitcher_impacts(self, multi_holdout_result: AblationResult) -> None:
        batter_keys = [k for k in multi_holdout_result.feature_impacts if k.startswith("batter:")]
        pitcher_keys = [k for k in multi_holdout_result.feature_impacts if k.startswith("pitcher:")]
        assert len(batter_keys) > 0
        assert len(pitcher_keys) > 0

    def test_multi_holdout_has_group_data(self, multi_holdout_result: AblationResult) -> None:
        assert isinstance(multi_holdout_result.group_impacts, dict)

    def test_multi_holdout_se_non_negative(self, multi_holdout_result: AblationResult) -> None:
        for se in multi_holdout_result.feature_standard_errors.values():
            assert se >= 0
        for se in multi_holdout_result.group_standard_errors.values():
            assert se >= 0

    def test_multi_holdout_with_validate(self) -> None:
        rows_by_season = {
            2021: _make_rows(10, 2021),
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2021: _make_pitcher_rows(10, 2021),
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            model_params={"multi_holdout": True, "validate": True, "n_repeats": 5},
        )
        result = model.ablate(config)
        assert isinstance(result, AblationResult)

    def test_multi_holdout_default_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_calls: list[str] = []
        original_cv = ablation_mod.compute_cv_permutation_importance

        def spy_cv(*args: Any, **kwargs: Any) -> Any:
            captured_calls.append("cv")
            return original_cv(*args, **kwargs)

        monkeypatch.setattr(ablation_mod, "compute_cv_permutation_importance", spy_cv)

        rows_by_season = {
            2022: _make_rows(10, 2022),
            2023: _make_rows(10, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(10, 2022),
            2023: _make_pitcher_rows(10, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2022, 2023])
        model.ablate(config)
        # CV function should NOT be called when multi_holdout is not set
        assert len(captured_calls) == 0


class TestSampleWeightTransformProperty:
    def test_preseason_default_transform_is_raw(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._sample_weight_transform == "raw"

    def test_live_model_transform_is_none(self) -> None:
        model = StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._sample_weight_transform is None


class TestTrainAppliesTransform:
    def test_train_applies_sqrt_transform(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_weights: list[list[float] | None] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_weights.append(kwargs.get("sample_weights"))
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
            model_params={"sample_weight_transform": "sqrt"},
        )
        model.train(config)
        # Batter weights should be sqrt of raw PA values
        assert len(captured_weights) == 2
        bat_weights = captured_weights[0]
        assert bat_weights is not None
        # All preseason rows have pa_1 = 1.0, so sqrt(1.0) = 1.0
        for w in bat_weights:
            assert math.isclose(w, math.sqrt(1.0), abs_tol=1e-9)

    def test_train_default_transform_is_raw(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_weights: list[list[float] | None] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_weights.append(kwargs.get("sample_weights"))
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
        )
        model.train(config)
        # Default "raw" transform should pass weights unchanged
        assert len(captured_weights) == 2
        bat_weights = captured_weights[0]
        assert bat_weights is not None
        # raw(1.0) = 1.0
        for w in bat_weights:
            assert w == 1.0


class TestTuneAppliesTransform:
    def test_tune_passes_transform_to_build_cv_folds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_transforms: list[Any] = []
        original_build = statcast_gbm_model_mod.build_cv_folds

        def spy_build(*args: Any, **kwargs: Any) -> Any:
            captured_transforms.append(kwargs.get("sample_weight_transform"))
            return original_build(*args, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "build_cv_folds", spy_build)

        rows_by_season = {
            2021: [_make_preseason_row(f"p_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2021: [_make_preseason_pitcher_row(f"pit_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            model_params={"param_grid": {"max_iter": [100]}, "sample_weight_transform": "sqrt"},
        )
        model.tune(config)
        # Called twice (batter + pitcher), both should have a transform function
        assert len(captured_transforms) == 2
        assert all(t is not None for t in captured_transforms)


class TestSweepMethod:
    @pytest.fixture(scope="class")
    def sweep_result(self) -> TuneResult:
        rows_by_season = {
            2021: [_make_preseason_row(f"p_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2021: [_make_preseason_pitcher_row(f"pit_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            model_params={"sweep_grid": {"sample_weight_transform": ["raw", "sqrt"]}},
        )
        return model.sweep(config)

    def test_returns_tune_result(self, sweep_result: TuneResult) -> None:
        assert isinstance(sweep_result, TuneResult)
        assert sweep_result.model_name == "statcast-gbm-preseason"

    def test_batter_params_contain_transform(self, sweep_result: TuneResult) -> None:
        assert "sample_weight_transform" in sweep_result.batter_params

    def test_pitcher_params_contain_transform(self, sweep_result: TuneResult) -> None:
        assert "sample_weight_transform" in sweep_result.pitcher_params

    def test_cv_rmse_has_entries(self, sweep_result: TuneResult) -> None:
        for target in BATTER_TARGETS:
            assert target in sweep_result.batter_cv_rmse
        for target in PITCHER_TARGETS:
            assert target in sweep_result.pitcher_cv_rmse


class TestPerTypeTransform:
    def test_train_applies_separate_batter_pitcher_transforms(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_weights: list[list[float] | None] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_weights.append(kwargs.get("sample_weights"))
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
            model_params={
                "batter": {"sample_weight_transform": "sqrt"},
                "pitcher": {"sample_weight_transform": "log1p"},
            },
        )
        model.train(config)
        assert len(captured_weights) == 2
        # Batter weights should be sqrt-transformed (pa_1=1.0 → sqrt(1.0)=1.0)
        bat_weights = captured_weights[0]
        assert bat_weights is not None
        for w in bat_weights:
            assert math.isclose(w, math.sqrt(1.0), abs_tol=1e-9)
        # Pitcher weights should be log1p-transformed (ip_1=1.0 → log1p(1.0)=ln(2))
        pit_weights = captured_weights[1]
        assert pit_weights is not None
        for w in pit_weights:
            assert math.isclose(w, math.log1p(1.0), abs_tol=1e-9)

    def test_train_falls_back_to_top_level_transform(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_weights: list[list[float] | None] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_weights.append(kwargs.get("sample_weights"))
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
            model_params={"sample_weight_transform": "sqrt"},
        )
        model.train(config)
        assert len(captured_weights) == 2
        # Both batter and pitcher should get sqrt transform
        for weights in captured_weights:
            assert weights is not None
            for w in weights:
                assert math.isclose(w, math.sqrt(1.0), abs_tol=1e-9)

    def test_train_per_type_overrides_top_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_weights: list[list[float] | None] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_weights.append(kwargs.get("sample_weights"))
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
            model_params={
                "sample_weight_transform": "sqrt",
                "batter": {"sample_weight_transform": "log1p"},
            },
        )
        model.train(config)
        assert len(captured_weights) == 2
        # Batter gets log1p (override)
        bat_weights = captured_weights[0]
        assert bat_weights is not None
        for w in bat_weights:
            assert math.isclose(w, math.log1p(1.0), abs_tol=1e-9)
        # Pitcher gets sqrt (top-level fallback)
        pit_weights = captured_weights[1]
        assert pit_weights is not None
        for w in pit_weights:
            assert math.isclose(w, math.sqrt(1.0), abs_tol=1e-9)

    def test_tune_applies_per_type_transforms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_transforms: list[Any] = []
        original_build = statcast_gbm_model_mod.build_cv_folds

        def spy_build(*args: Any, **kwargs: Any) -> Any:
            captured_transforms.append(kwargs.get("sample_weight_transform"))
            return original_build(*args, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "build_cv_folds", spy_build)

        rows_by_season = {
            2021: [_make_preseason_row(f"p_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2021: [_make_preseason_pitcher_row(f"pit_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            model_params={
                "param_grid": {"max_iter": [100]},
                "batter": {"sample_weight_transform": "sqrt"},
                "pitcher": {"sample_weight_transform": "log1p"},
            },
        )
        model.tune(config)
        assert len(captured_transforms) == 2
        # Both should be callable (not None)
        assert all(t is not None for t in captured_transforms)
        # Verify they're different transforms by applying to a test value
        test_val = [100.0]
        batter_result = captured_transforms[0](test_val)
        pitcher_result = captured_transforms[1](test_val)
        assert math.isclose(batter_result[0], math.sqrt(100.0), abs_tol=1e-9)
        assert math.isclose(pitcher_result[0], math.log1p(100.0), abs_tol=1e-9)


class TestConfigTopPassThrough:
    def test_sweep_passes_config_top_to_sweep_cv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_kwargs: list[dict[str, Any]] = []
        original_sweep_cv = statcast_gbm_model_mod.sweep_cv

        def spy_sweep_cv(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return original_sweep_cv(*args, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "sweep_cv", spy_sweep_cv)

        rows_by_season = {
            2021: [_make_preseason_row(f"p_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2021: [_make_preseason_pitcher_row(f"pit_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            top=100,
            model_params={"sweep_grid": {"sample_weight_transform": ["raw", "sqrt"]}},
        )
        model.sweep(config)
        assert len(captured_kwargs) == 2
        assert captured_kwargs[0]["test_top_n"] == 100
        assert captured_kwargs[1]["test_top_n"] == 100

    def test_tune_passes_config_top_to_build_cv_folds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_kwargs: list[dict[str, Any]] = []
        original_build = statcast_gbm_model_mod.build_cv_folds

        def spy_build(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return original_build(*args, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "build_cv_folds", spy_build)

        rows_by_season = {
            2021: [_make_preseason_row(f"p_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2021: [_make_preseason_pitcher_row(f"pit_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            top=100,
            model_params={"param_grid": {"max_iter": [100]}},
        )
        model.tune(config)
        assert len(captured_kwargs) == 2
        assert captured_kwargs[0]["test_top_n"] == 100
        assert captured_kwargs[1]["test_top_n"] == 100


class TestWarRankColumn:
    def test_sweep_passes_war_as_test_rank_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_kwargs: list[dict[str, Any]] = []
        original_sweep_cv = statcast_gbm_model_mod.sweep_cv

        def spy_sweep_cv(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return original_sweep_cv(*args, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "sweep_cv", spy_sweep_cv)

        rows_by_season = {
            2021: [_make_preseason_row(f"p_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2021: [_make_preseason_pitcher_row(f"pit_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            top=100,
            model_params={"sweep_grid": {"sample_weight_transform": ["raw", "sqrt"]}},
        )
        model.sweep(config)
        assert len(captured_kwargs) == 2
        assert captured_kwargs[0]["test_rank_column"] == "war"
        assert captured_kwargs[1]["test_rank_column"] == "war"

    def test_tune_passes_war_as_test_rank_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_kwargs: list[dict[str, Any]] = []
        original_build = statcast_gbm_model_mod.build_cv_folds

        def spy_build(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return original_build(*args, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "build_cv_folds", spy_build)

        rows_by_season = {
            2021: [_make_preseason_row(f"p_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2021: [_make_preseason_pitcher_row(f"pit_{i}", 2021) for i in range(10)],
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            top=100,
            model_params={"param_grid": {"max_iter": [100]}},
        )
        model.tune(config)
        assert len(captured_kwargs) == 2
        assert captured_kwargs[0]["test_rank_column"] == "war"
        assert captured_kwargs[1]["test_rank_column"] == "war"


class TestSweepSupportedOperations:
    def test_preseason_supports_sweep(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert "sweep" in model.supported_operations

    def test_live_does_not_support_sweep(self) -> None:
        model = StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert "sweep" not in model.supported_operations


class TestMinActivityProperties:
    def test_preseason_batter_min_pa_default(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._batter_min_pa == 100

    def test_preseason_pitcher_min_ip_default(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._pitcher_min_ip == 20

    def test_live_model_min_pa_is_zero(self) -> None:
        model = StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._batter_min_pa == 0

    def test_live_model_min_ip_is_zero(self) -> None:
        model = StatcastGBMModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._pitcher_min_ip == 0


class TestResolveMinActivity:
    def test_resolve_min_pa_returns_property_default(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._resolve_min_pa({}) == 100

    def test_resolve_min_pa_overridden_by_model_params(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._resolve_min_pa({"min_pa": 50}) == 50

    def test_resolve_min_ip_returns_property_default(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._resolve_min_ip({}) == 20

    def test_resolve_min_ip_overridden_by_model_params(self) -> None:
        model = StatcastGBMPreseasonModel(assembler=_NULL_ASSEMBLER, evaluator=_NULL_EVALUATOR)
        assert model._resolve_min_ip({"min_ip": 10}) == 10


class TestMinActivityTrainFilter:
    def test_train_filters_low_pa_batters(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_X: list[list[list[float]]] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_X.append(X)
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        high_pa_rows = [_make_preseason_row(f"high_{i}", 2022) for i in range(5)]
        for r in high_pa_rows:
            r["pa"] = 500
        low_pa_rows = [_make_preseason_row(f"low_{i}", 2022) for i in range(5)]
        for r in low_pa_rows:
            r["pa"] = 50

        rows_by_season = {
            2022: high_pa_rows + low_pa_rows,
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2022, 2023], artifacts_dir=str(tmp_path))
        model.train(config)
        # First fit_models call is batters — train on 2022, holdout on 2023
        # Only the 5 high-PA rows from 2022 should be in training
        bat_X_train = captured_X[0]
        assert len(bat_X_train) == 5

    def test_train_filters_low_ip_pitchers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_X: list[list[list[float]]] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_X.append(X)
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        high_ip_rows = [_make_preseason_pitcher_row(f"high_{i}", 2022) for i in range(5)]
        for r in high_ip_rows:
            r["ip"] = 180.0
        low_ip_rows = [_make_preseason_pitcher_row(f"low_{i}", 2022) for i in range(5)]
        for r in low_ip_rows:
            r["ip"] = 10.0

        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: high_ip_rows + low_ip_rows,
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2022, 2023], artifacts_dir=str(tmp_path))
        model.train(config)
        # Second fit_models call is pitchers
        pit_X_train = captured_X[1]
        assert len(pit_X_train) == 5

    def test_train_holdout_not_filtered(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_holdout_X: list[list[list[float]]] = []
        original_score = statcast_gbm_model_mod.score_predictions

        def spy_score(models: Any, X: Any, y: Any) -> Any:
            captured_holdout_X.append(X)
            return original_score(models, X, y)

        monkeypatch.setattr(statcast_gbm_model_mod, "score_predictions", spy_score)

        rows_by_season = {
            2022: [_make_preseason_row(f"p_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(5)]
            + [{**_make_preseason_row(f"low_{i}", 2023), "pa": 50} for i in range(5)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(seasons=[2022, 2023], artifacts_dir=str(tmp_path))
        model.train(config)
        # Holdout is 2023 with 10 batter rows (5 high + 5 low PA) — all should be present
        bat_holdout_X = captured_holdout_X[0]
        assert len(bat_holdout_X) == 10

    def test_train_filter_configurable_via_model_params(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_X: list[list[list[float]]] = []
        original_fit = statcast_gbm_model_mod.fit_models

        def spy_fit(X: Any, y: Any, params: dict[str, Any], **kwargs: Any) -> Any:
            captured_X.append(X)
            return original_fit(X, y, params, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "fit_models", spy_fit)

        rows = []
        for i in range(10):
            r = _make_preseason_row(f"p_{i}", 2022)
            r["pa"] = 100 + i * 50  # 100, 150, 200, ..., 550
            rows.append(r)

        rows_by_season = {
            2022: rows,
            2023: [_make_preseason_row(f"p_{i}", 2023) for i in range(10)],
        }
        pitcher_rows = {
            2022: [_make_preseason_pitcher_row(f"pit_{i}", 2022) for i in range(10)],
            2023: [_make_preseason_pitcher_row(f"pit_{i}", 2023) for i in range(10)],
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2022, 2023],
            artifacts_dir=str(tmp_path),
            model_params={"batter": {"min_pa": 200}},
        )
        model.train(config)
        # PA values: 100, 150, 200, 250, 300, 350, 400, 450, 500, 550
        # With min_pa=200, rows with pa >= 200 → 8 rows
        bat_X_train = captured_X[0]
        assert len(bat_X_train) == 8


class TestMinActivityTuneFilter:
    def test_tune_passes_filtered_rows_to_build_cv_folds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_rows: list[list[dict[str, Any]]] = []
        original_build = statcast_gbm_model_mod.build_cv_folds

        def spy_build(all_rows: list[dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
            captured_rows.append(all_rows)
            return original_build(all_rows, *args, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "build_cv_folds", spy_build)

        high_pa = [_make_preseason_row(f"high_{i}", s) for s in (2021, 2022, 2023) for i in range(5)]
        for r in high_pa:
            r["pa"] = 500
        low_pa = [_make_preseason_row(f"low_{i}", s) for s in (2021, 2022, 2023) for i in range(5)]
        for r in low_pa:
            r["pa"] = 50

        rows_by_season: dict[int, list[dict[str, Any]]] = {}
        for s in (2021, 2022, 2023):
            rows_by_season[s] = [r for r in high_pa + low_pa if r["season"] == s]

        pitcher_rows = {s: [_make_preseason_pitcher_row(f"pit_{i}", s) for i in range(10)] for s in (2021, 2022, 2023)}
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            model_params={"param_grid": {"max_iter": [100]}},
        )
        model.tune(config)
        # build_cv_folds called for batters first — should have only high-PA rows
        # 3 seasons × 5 high-PA rows = 15 (not 30)
        bat_rows = captured_rows[0]
        assert len(bat_rows) == 15


class TestMinActivitySweepFilter:
    def test_sweep_passes_filtered_rows_to_sweep_cv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_rows: list[list[dict[str, Any]]] = []
        original_sweep = statcast_gbm_model_mod.sweep_cv

        def spy_sweep(all_rows: list[dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
            captured_rows.append(all_rows)
            return original_sweep(all_rows, *args, **kwargs)

        monkeypatch.setattr(statcast_gbm_model_mod, "sweep_cv", spy_sweep)

        high_pa = [_make_preseason_row(f"high_{i}", s) for s in (2021, 2022, 2023) for i in range(5)]
        for r in high_pa:
            r["pa"] = 500
        low_pa = [_make_preseason_row(f"low_{i}", s) for s in (2021, 2022, 2023) for i in range(5)]
        for r in low_pa:
            r["pa"] = 50

        rows_by_season: dict[int, list[dict[str, Any]]] = {}
        for s in (2021, 2022, 2023):
            rows_by_season[s] = [r for r in high_pa + low_pa if r["season"] == s]

        pitcher_rows = {s: [_make_preseason_pitcher_row(f"pit_{i}", s) for i in range(10)] for s in (2021, 2022, 2023)}
        assembler = FakeAssembler(rows_by_season, pitcher_rows)
        model = StatcastGBMPreseasonModel(assembler=assembler, evaluator=_NULL_EVALUATOR)
        config = ModelConfig(
            seasons=[2021, 2022, 2023],
            model_params={"sweep_grid": {"sample_weight_transform": ["raw", "sqrt"]}},
        )
        model.sweep(config)
        # sweep_cv called for batters first — should have only high-PA rows
        bat_rows = captured_rows[0]
        assert len(bat_rows) == 15
