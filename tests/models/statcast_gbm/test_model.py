from pathlib import Path
from typing import Any

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
)
from fantasy_baseball_manager.models.statcast_gbm.features import batter_feature_columns, pitcher_feature_columns
from fantasy_baseball_manager.models.statcast_gbm.model import StatcastGBMModel
from fantasy_baseball_manager.models.statcast_gbm.targets import BATTER_TARGETS, PITCHER_TARGETS

_FEATURE_COLUMNS = batter_feature_columns()
_PITCHER_FEATURE_COLUMNS = pitcher_feature_columns()


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


class TestStatcastGBMProtocol:
    def test_is_model(self) -> None:
        assert isinstance(StatcastGBMModel(), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(StatcastGBMModel(), Preparable)

    def test_is_trainable(self) -> None:
        assert isinstance(StatcastGBMModel(), Trainable)

    def test_is_evaluable(self) -> None:
        assert isinstance(StatcastGBMModel(), Evaluable)

    def test_is_predictable(self) -> None:
        assert isinstance(StatcastGBMModel(), Predictable)

    def test_is_ablatable(self) -> None:
        assert isinstance(StatcastGBMModel(), Ablatable)

    def test_is_feature_introspectable(self) -> None:
        assert isinstance(StatcastGBMModel(), FeatureIntrospectable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(StatcastGBMModel(), FineTunable)

    def test_name(self) -> None:
        assert StatcastGBMModel().name == "statcast-gbm"

    def test_artifact_type(self) -> None:
        assert StatcastGBMModel().artifact_type == "file"

    def test_supported_operations(self) -> None:
        ops = StatcastGBMModel().supported_operations
        assert ops == frozenset({"prepare", "train", "evaluate", "predict", "ablate"})

    def test_declared_features_not_empty(self) -> None:
        features = StatcastGBMModel().declared_features
        assert len(features) > 0


class TestStatcastGBMPrepare:
    def test_prepare_returns_result(self) -> None:
        assembler = FakeAssembler()
        model = StatcastGBMModel(assembler=assembler)
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
        model = StatcastGBMModel(assembler=assembler)
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
        model = StatcastGBMModel(assembler=assembler)
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
        model = StatcastGBMModel(assembler=assembler)
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
        model = StatcastGBMModel(assembler=assembler)
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
        model = StatcastGBMModel(assembler=assembler)
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
        model = StatcastGBMModel(evaluator=evaluator)
        config = ModelConfig(seasons=[2023], version="v1")
        result = model.evaluate(config)
        assert isinstance(result, SystemMetrics)
        assert result.system == "statcast-gbm"
        assert evaluator.called_with["system"] == "statcast-gbm"
        assert evaluator.called_with["version"] == "v1"
        assert evaluator.called_with["season"] == 2023


class TestStatcastGBMAblate:
    def _build_model_and_config(self) -> tuple[StatcastGBMModel, ModelConfig]:
        rows_by_season = {
            2022: _make_rows(30, 2022),
            2023: _make_rows(30, 2023),
        }
        pitcher_rows_by_season = {
            2022: _make_pitcher_rows(30, 2022),
            2023: _make_pitcher_rows(30, 2023),
        }
        assembler = FakeAssembler(rows_by_season, pitcher_rows_by_season)
        model = StatcastGBMModel(assembler=assembler)
        config = ModelConfig(seasons=[2022, 2023])
        return model, config

    def test_ablate_returns_result(self) -> None:
        model, config = self._build_model_and_config()
        result = model.ablate(config)
        assert isinstance(result, AblationResult)
        assert result.model_name == "statcast-gbm"
        assert isinstance(result.feature_impacts, dict)

    def test_ablate_returns_nonempty_impacts(self) -> None:
        model, config = self._build_model_and_config()
        result = model.ablate(config)
        assert len(result.feature_impacts) > 0

    def test_ablate_impacts_include_batter_features(self) -> None:
        model, config = self._build_model_and_config()
        result = model.ablate(config)
        batter_keys = [k for k in result.feature_impacts if k.startswith("batter:")]
        assert len(batter_keys) > 0

    def test_ablate_impacts_include_pitcher_features(self) -> None:
        model, config = self._build_model_and_config()
        result = model.ablate(config)
        pitcher_keys = [k for k in result.feature_impacts if k.startswith("pitcher:")]
        assert len(pitcher_keys) > 0
