from pathlib import Path
from typing import Any

import pytest

import fantasy_baseball_manager.features.group_library  # noqa: F401 — trigger registration
from fantasy_baseball_manager.features.groups import FeatureGroup, get_group, list_groups
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
import fantasy_baseball_manager.models.composite  # noqa: F401 — trigger alias registration
from fantasy_baseball_manager.models.composite.engine import EngineConfig, GBMEngine, MarcelEngine
from fantasy_baseball_manager.models.composite.features import (
    build_composite_batting_features,
    build_composite_pitching_features,
    feature_columns,
)
from fantasy_baseball_manager.models.composite.model import (
    DEFAULT_GROUPS,
    CompositeModel,
    _build_marcel_config,
    _resolve_group,
)
from fantasy_baseball_manager.models.marcel.types import MarcelConfig
from fantasy_baseball_manager.domain.evaluation import StatMetrics, SystemMetrics
from fantasy_baseball_manager.models.protocols import (
    Ablatable,
    AblationResult,
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    Preparable,
    TrainResult,
    Trainable,
)
from fantasy_baseball_manager.models.registry import _clear, get, register, register_alias

# Snapshot groups at import time — before any test can clear the global registry.
_GROUPS: dict[str, FeatureGroup] = {name: get_group(name) for name in list_groups()}


def _test_lookup(name: str) -> FeatureGroup:
    """Local group lookup isolated from global registry mutations."""
    if name not in _GROUPS:
        raise KeyError(f"'{name}': no feature group in test snapshot")
    return _GROUPS[name]


class FakeAssembler:
    """In-memory assembler for testing predict()."""

    def __init__(self, batting_rows: list[dict[str, Any]], pitching_rows: list[dict[str, Any]] | None = None) -> None:
        self._batting_rows = batting_rows
        self._pitching_rows = pitching_rows or []
        self._next_id = 1

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.get_or_materialize(feature_set)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        if "pitching" in feature_set.name:
            rows = self._pitching_rows
        else:
            rows = self._batting_rows
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
        return DatasetSplits(train=handle, validation=None, holdout=None)

    def read(self, handle: DatasetHandle) -> list[dict[str, Any]]:
        if "pitching" in handle.table_name:
            return self._pitching_rows
        return self._batting_rows


_NULL_ASSEMBLER = FakeAssembler(batting_rows=[])


class FakeEvaluator:
    """Records calls and returns a canned SystemMetrics."""

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


class TestCompositeModelProtocol:
    def test_is_model(self) -> None:
        assert isinstance(CompositeModel(assembler=_NULL_ASSEMBLER), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(CompositeModel(assembler=_NULL_ASSEMBLER), Preparable)

    def test_is_predictable(self) -> None:
        assert isinstance(CompositeModel(assembler=_NULL_ASSEMBLER), Predictable)

    def test_is_trainable(self) -> None:
        assert isinstance(CompositeModel(assembler=_NULL_ASSEMBLER), Trainable)

    def test_is_evaluable(self) -> None:
        assert isinstance(CompositeModel(assembler=_NULL_ASSEMBLER, evaluator=FakeEvaluator()), Evaluable)

    def test_is_ablatable(self) -> None:
        assert isinstance(CompositeModel(assembler=_NULL_ASSEMBLER), Ablatable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(CompositeModel(assembler=_NULL_ASSEMBLER), FineTunable)

    def test_supported_operations_includes_evaluate_with_evaluator(self) -> None:
        model = CompositeModel(assembler=_NULL_ASSEMBLER, evaluator=FakeEvaluator())
        assert "evaluate" in model.supported_operations

    def test_supported_operations_excludes_evaluate_without_evaluator(self) -> None:
        model = CompositeModel(assembler=_NULL_ASSEMBLER)
        assert "evaluate" not in model.supported_operations

    def test_supported_operations_includes_ablate_with_gbm_engine(self) -> None:
        model = CompositeModel(assembler=_NULL_ASSEMBLER, engine=GBMEngine())
        assert "ablate" in model.supported_operations

    def test_supported_operations_excludes_ablate_with_marcel_engine(self) -> None:
        model = CompositeModel(assembler=_NULL_ASSEMBLER, engine=MarcelEngine())
        assert "ablate" not in model.supported_operations

    def test_supported_operations_gbm_with_evaluator(self) -> None:
        model = CompositeModel(assembler=_NULL_ASSEMBLER, engine=GBMEngine(), evaluator=FakeEvaluator())
        assert model.supported_operations == frozenset({"prepare", "train", "predict", "evaluate", "ablate"})

    def test_name(self) -> None:
        assert CompositeModel(assembler=_NULL_ASSEMBLER).name == "composite"

    def test_name_uses_model_name_param(self) -> None:
        assert CompositeModel(assembler=_NULL_ASSEMBLER, model_name="composite-mle").name == "composite-mle"

    def test_supported_operations(self) -> None:
        assert CompositeModel(assembler=_NULL_ASSEMBLER).supported_operations == frozenset({"prepare", "predict"})

    def test_artifact_type(self) -> None:
        assert CompositeModel(assembler=_NULL_ASSEMBLER).artifact_type == "none"


class TestCompositeEvaluate:
    def test_evaluate_delegates_to_evaluator(self) -> None:
        evaluator = FakeEvaluator()
        model = CompositeModel(assembler=_NULL_ASSEMBLER, evaluator=evaluator)
        config = ModelConfig(seasons=[2023], version="v1")
        result = model.evaluate(config)
        assert isinstance(result, SystemMetrics)
        assert evaluator.calls == [("composite", "v1", 2023, None)]

    def test_evaluate_uses_config_version(self) -> None:
        evaluator = FakeEvaluator()
        model = CompositeModel(assembler=_NULL_ASSEMBLER, evaluator=evaluator)
        config = ModelConfig(seasons=[2023], version="v2")
        model.evaluate(config)
        assert evaluator.calls[0][1] == "v2"

    def test_evaluate_passes_top(self) -> None:
        evaluator = FakeEvaluator()
        model = CompositeModel(assembler=_NULL_ASSEMBLER, evaluator=evaluator)
        config = ModelConfig(seasons=[2023], top=50)
        model.evaluate(config)
        assert evaluator.calls[0][3] == 50

    def test_evaluate_uses_model_name(self) -> None:
        evaluator = FakeEvaluator()
        model = CompositeModel(assembler=_NULL_ASSEMBLER, model_name="composite-mle", evaluator=evaluator)
        config = ModelConfig(seasons=[2023])
        model.evaluate(config)
        assert evaluator.calls[0][0] == "composite-mle"

    def test_evaluate_without_evaluator_raises(self) -> None:
        model = CompositeModel(assembler=_NULL_ASSEMBLER)
        config = ModelConfig(seasons=[2023])
        with pytest.raises(TypeError):
            model.evaluate(config)


class TestResolveGroup:
    def test_resolve_group_static(self) -> None:
        config = MarcelConfig()
        group = _resolve_group("age", config, lookup=_test_lookup)
        assert group.name == "age"
        assert group.player_type == "both"

    def test_resolve_group_batting_counting_lags(self) -> None:
        config = MarcelConfig(batting_categories=("hr", "rbi"), batting_weights=(5.0, 4.0))
        group = _resolve_group("batting_counting_lags", config, lookup=_test_lookup)
        assert group.name == "batting_counting_lags"
        assert group.player_type == "batter"
        # 2 lags × (pa + hr + rbi) = 6 features
        assert len(group.features) == 6

    def test_resolve_group_pitching_counting_lags(self) -> None:
        config = MarcelConfig(pitching_categories=("so", "bb"), pitching_weights=(5.0, 4.0))
        group = _resolve_group("pitching_counting_lags", config, lookup=_test_lookup)
        assert group.name == "pitching_counting_lags"
        assert group.player_type == "pitcher"
        # 2 lags × (ip + g + gs + so + bb) = 10 features
        assert len(group.features) == 10

    def test_resolve_group_unknown_raises(self) -> None:
        config = MarcelConfig()
        with pytest.raises(KeyError):
            _resolve_group("nonexistent_group_xyz", config, lookup=_test_lookup)

    def test_default_groups_constant(self) -> None:
        assert DEFAULT_GROUPS == (
            "age",
            "projected_batting_pt",
            "projected_pitching_pt",
            "batting_counting_lags",
            "pitching_counting_lags",
        )


class TestBuildFeatureSets:
    def test_build_feature_sets_default_groups(self) -> None:
        """With no feature_groups in config, batting FeatureSet contains age, proj_pa, counting lags, transforms."""
        assembler = FakeAssembler(batting_rows=[], pitching_rows=[])
        model = CompositeModel(assembler=assembler, group_lookup=_test_lookup)
        config = ModelConfig(seasons=[2023], model_params={"batting_categories": ["hr"]})
        batting_fs, pitching_fs = model._build_feature_sets(_build_marcel_config(config.model_params), config)
        bat_names = {f.name for f in batting_fs.features}
        # Must include age, proj_pa, counting lags, weighted_rates, league_averages
        assert "age" in bat_names
        assert "proj_pa" in bat_names
        assert "hr_1" in bat_names
        assert "pa_1" in bat_names
        assert "batting_weighted_rates" in bat_names
        assert "batting_league_averages" in bat_names

    def test_build_feature_sets_with_mle_groups(self) -> None:
        """With feature_groups including mle_batter_rates, batting FeatureSet contains MLE features."""
        assembler = FakeAssembler(batting_rows=[], pitching_rows=[])
        model = CompositeModel(assembler=assembler, group_lookup=_test_lookup)
        config = ModelConfig(
            seasons=[2023],
            model_params={
                "batting_categories": ["hr"],
                "feature_groups": [
                    "age",
                    "projected_batting_pt",
                    "projected_pitching_pt",
                    "batting_counting_lags",
                    "pitching_counting_lags",
                    "mle_batter_rates",
                ],
            },
        )
        batting_fs, pitching_fs = model._build_feature_sets(_build_marcel_config(config.model_params), config)
        bat_names = {f.name for f in batting_fs.features}
        assert "mle_avg" in bat_names
        assert "mle_obp" in bat_names
        assert "mle_slg" in bat_names

    def test_build_feature_sets_naming(self) -> None:
        """Feature set names contain batting/pitching for FakeAssembler routing."""
        assembler = FakeAssembler(batting_rows=[], pitching_rows=[])
        model = CompositeModel(assembler=assembler, group_lookup=_test_lookup)
        config = ModelConfig(seasons=[2023], model_params={"batting_categories": ["hr"]})
        batting_fs, pitching_fs = model._build_feature_sets(_build_marcel_config(config.model_params), config)
        assert "batting" in batting_fs.name
        assert "pitching" in pitching_fs.name


class TestDefaultGroupsRegression:
    """Verify new group-based feature building produces the same features as the old inline builders."""

    def test_default_batting_features_match_inline(self) -> None:
        marcel_config = MarcelConfig(batting_categories=("hr", "rbi", "sb"))
        config = ModelConfig(seasons=[2023], model_params={"batting_categories": ["hr", "rbi", "sb"]})
        model = CompositeModel(assembler=FakeAssembler([]), group_lookup=_test_lookup)
        batting_fs, _ = model._build_feature_sets(marcel_config, config)
        new_names = {f.name for f in batting_fs.features}

        old_features = build_composite_batting_features(marcel_config.batting_categories, marcel_config.batting_weights)
        old_names = {f.name for f in old_features}
        assert new_names == old_names

    def test_default_pitching_features_match_inline(self) -> None:
        marcel_config = MarcelConfig(pitching_categories=("so", "bb", "era"))
        config = ModelConfig(seasons=[2023], model_params={"pitching_categories": ["so", "bb", "era"]})
        model = CompositeModel(assembler=FakeAssembler([]), group_lookup=_test_lookup)
        _, pitching_fs = model._build_feature_sets(marcel_config, config)
        new_names = {f.name for f in pitching_fs.features}

        old_features = build_composite_pitching_features(
            marcel_config.pitching_categories, marcel_config.pitching_weights
        )
        old_names = {f.name for f in old_features}
        assert new_names == old_names


class TestCompositeAliases:
    def test_aliases_registered(self) -> None:
        _clear()
        register("composite")(CompositeModel)
        for alias in ("composite-mle", "composite-statcast", "composite-full"):
            register_alias(alias, "composite")
        for alias in ("composite-mle", "composite-statcast", "composite-full"):
            assert get(alias) is CompositeModel


class TestCompositePredict:
    def test_predict_batter_returns_counting_stats(self) -> None:
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "proj_pa": 555,
                "pa_1": 600,
                "pa_2": 550,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        result = CompositeModel(assembler=assembler, group_lookup=_test_lookup).predict(config)
        assert result.model_name == "composite"
        assert len(result.predictions) == 1
        pred = result.predictions[0]
        assert pred["player_id"] == 1
        assert "hr" in pred

    def test_predict_uses_projected_pt_not_internal(self) -> None:
        """Counting stats must use proj_pa from the playing-time model, NOT internally computed PT."""
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "proj_pa": 400,  # lower than internal would compute
                "pa_1": 600,
                "pa_2": 550,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        result = CompositeModel(assembler=assembler, group_lookup=_test_lookup).predict(config)
        pred = result.predictions[0]
        # Internal PT would be 200 + 0.5*600 + 0.1*550 = 555
        # But we should use proj_pa=400
        # hr = rate * 400, which should be less than rate * 555
        # The rate comes from regress_to_mean + age_adjust
        # Just verify the pa in the output matches 400
        assert pred["pa"] == 400

    def test_predict_pitcher_uses_projected_ip(self) -> None:
        pitching_rows = [
            {
                "player_id": 10,
                "season": 2023,
                "age": 28,
                "proj_ip": 150.0,
                "ip_1": 180.0,
                "ip_2": 170.0,
                "g_1": 30,
                "g_2": 28,
                "gs_1": 30,
                "gs_2": 28,
                "so_1": 200.0,
                "so_2": 180.0,
                "so_wavg": 1110.0 / 1040.0,
                "weighted_pt": 1040.0,
                "league_so_rate": 200.0 / 180.0,
            },
        ]
        assembler = FakeAssembler(batting_rows=[], pitching_rows=pitching_rows)
        config = ModelConfig(
            seasons=[2023],
            model_params={
                "batting_categories": ["hr"],
                "pitching_categories": ["so"],
            },
        )
        result = CompositeModel(assembler=assembler, group_lookup=_test_lookup).predict(config)
        pitcher_preds = [p for p in result.predictions if p.get("player_type") == "pitcher"]
        assert len(pitcher_preds) == 1
        assert pitcher_preds[0]["ip"] == 150.0

    def test_predict_empty_data(self) -> None:
        assembler = FakeAssembler(batting_rows=[], pitching_rows=[])
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        result = CompositeModel(assembler=assembler, group_lookup=_test_lookup).predict(config)
        assert result.model_name == "composite"
        assert len(result.predictions) == 0

    def test_predict_counting_stats_are_rate_times_pt(self) -> None:
        """Verify the critical invariant: counting = rate * proj_pa."""
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "proj_pa": 600,
                "pa_1": 600,
                "pa_2": 550,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        result = CompositeModel(assembler=assembler, group_lookup=_test_lookup).predict(config)
        pred = result.predictions[0]
        # rates dict should be present in the raw prediction data
        assert pred["rates"]["hr"] * 600 == pred["hr"]

    def test_predict_uses_model_name(self) -> None:
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "proj_pa": 555,
                "pa_1": 600,
                "pa_2": 550,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(seasons=[2023], model_params={"batting_categories": ["hr"]})
        result = CompositeModel(assembler=assembler, model_name="composite-mle", group_lookup=_test_lookup).predict(
            config
        )
        assert result.model_name == "composite-mle"

    def test_predict_projected_season(self) -> None:
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "proj_pa": 555,
                "pa_1": 600,
                "pa_2": 550,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        assembler = FakeAssembler(batting_rows)
        config = ModelConfig(seasons=[2023], model_params={"batting_categories": ["hr"]})
        result = CompositeModel(assembler=assembler, group_lookup=_test_lookup).predict(config)
        assert result.predictions[0]["season"] == 2024


class FakeEngine:
    """Fake engine that records calls for delegation testing."""

    def __init__(
        self,
        ops: frozenset[str] = frozenset({"custom_op"}),
        artifact: str = "custom_artifact",
    ) -> None:
        self._ops = ops
        self._artifact = artifact
        self.calls: list[
            tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, float], dict[int, float], EngineConfig]
        ] = []

    @property
    def supported_operations(self) -> frozenset[str]:
        return self._ops

    @property
    def artifact_type(self) -> str:
        return self._artifact

    def predict(
        self,
        bat_rows: list[dict[str, Any]],
        pitch_rows: list[dict[str, Any]],
        bat_pt: dict[int, float],
        pitch_pt: dict[int, float],
        config: EngineConfig,
    ) -> list[dict[str, Any]]:
        self.calls.append((bat_rows, pitch_rows, bat_pt, pitch_pt, config))
        return [{"player_id": 99, "season": config.projected_season, "player_type": "batter"}]


class TestEngineDelegation:
    def test_supported_operations_delegates_to_engine(self) -> None:
        engine = FakeEngine(ops=frozenset({"train", "predict", "evaluate"}))
        model = CompositeModel(assembler=_NULL_ASSEMBLER, engine=engine)
        assert model.supported_operations == frozenset({"train", "predict", "evaluate"})

    def test_artifact_type_delegates_to_engine(self) -> None:
        engine = FakeEngine(artifact="directory")
        model = CompositeModel(assembler=_NULL_ASSEMBLER, engine=engine)
        assert model.artifact_type == "directory"

    def test_predict_delegates_to_engine(self) -> None:
        batting_rows = [
            {
                "player_id": 1,
                "season": 2023,
                "age": 29,
                "proj_pa": 555,
                "pa_1": 600,
                "pa_2": 550,
                "hr_1": 30.0,
                "hr_2": 25.0,
                "hr_wavg": 310.0 / 6700.0,
                "weighted_pt": 6700.0,
                "league_hr_rate": 50.0 / 1100.0,
            },
        ]
        engine = FakeEngine()
        assembler = FakeAssembler(batting_rows)
        model = CompositeModel(assembler=assembler, group_lookup=_test_lookup, engine=engine)
        config = ModelConfig(seasons=[2023], model_params={"batting_categories": ["hr"]})
        result = model.predict(config)

        # Engine was called exactly once
        assert len(engine.calls) == 1
        bat_rows_arg, pitch_rows_arg, bat_pt_arg, pitch_pt_arg, engine_config = engine.calls[0]

        # Rows passed through
        assert len(bat_rows_arg) == 1
        assert bat_rows_arg[0]["player_id"] == 1

        # PT extracted correctly
        assert bat_pt_arg == {1: 555.0}

        # EngineConfig populated
        assert isinstance(engine_config, EngineConfig)
        assert engine_config.projected_season == 2024
        assert engine_config.version == "latest"
        assert engine_config.system_name == "composite"

        # Result uses engine output
        assert result.predictions == [{"player_id": 99, "season": 2024, "player_type": "batter"}]


class SeasonAwareFakeAssembler:
    """Assembler that routes rows by season for train/holdout splits."""

    def __init__(
        self,
        rows_by_season: dict[int, list[dict[str, Any]]] | None = None,
        pitcher_rows_by_season: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._next_id = 1
        self._rows_by_season = rows_by_season or {}
        self._pitcher_rows_by_season = pitcher_rows_by_season or {}

    def _select_rows(self, feature_set_name: str) -> dict[int, list[dict[str, Any]]]:
        if "pitching" in feature_set_name:
            return self._pitcher_rows_by_season
        return self._rows_by_season

    def materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        return self.get_or_materialize(feature_set)

    def get_or_materialize(self, feature_set: FeatureSet) -> DatasetHandle:
        source = self._select_rows(feature_set.name)
        all_rows: list[dict[str, Any]] = []
        for s in feature_set.seasons:
            all_rows.extend(source.get(s, []))
        handle = DatasetHandle(
            dataset_id=self._next_id,
            feature_set_id=self._next_id,
            table_name=f"ds_{feature_set.name}",
            row_count=len(all_rows),
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
        source = self._select_rows(handle.table_name)
        train_seasons = list(train)
        holdout_seasons = holdout or []
        train_rows: list[dict[str, Any]] = []
        for s in train_seasons:
            train_rows.extend(source.get(s, []))
        holdout_rows: list[dict[str, Any]] = []
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


def _make_batter_row(player_id: int, season: int) -> dict[str, Any]:
    """Batter row with default composite features (batting_categories=["hr"]) + targets."""
    return {
        "player_id": player_id,
        "season": season,
        "age": 28,
        "proj_pa": 550,
        "pa_1": 600,
        "pa_2": 550,
        "hr_1": 25.0,
        "hr_2": 22.0,
        "hr_wavg": 0.04,
        "weighted_pt": 6500.0,
        "league_hr_rate": 0.03,
        # Targets
        "target_avg": 0.270,
        "target_obp": 0.340,
        "target_slg": 0.440,
        "target_woba": 0.330,
        "target_h": 145,
        "target_hr": 24,
        "target_ab": 500,
        "target_so": 110,
        "target_sf": 4,
    }


def _make_pitcher_row(player_id: int, season: int) -> dict[str, Any]:
    """Pitcher row with default composite features (pitching_categories=["so"]) + targets."""
    return {
        "player_id": player_id,
        "season": season,
        "age": 27,
        "proj_ip": 160.0,
        "ip_1": 175.0,
        "ip_2": 165.0,
        "g_1": 32,
        "g_2": 30,
        "gs_1": 32,
        "gs_2": 30,
        "so_1": 190.0,
        "so_2": 175.0,
        "so_wavg": 1.05,
        "weighted_pt": 1000.0,
        "league_so_rate": 1.1,
        # Targets
        "target_era": 3.60,
        "target_fip": 3.50,
        "target_k_per_9": 9.5,
        "target_bb_per_9": 2.8,
        "target_whip": 1.18,
        "target_h": 155,
        "target_hr": 18,
        "target_ip": 175.0,
        "target_so": 190,
    }


class RecordingGBMEngine(GBMEngine):
    """GBMEngine subclass that records train()/predict() args instead of fitting real models."""

    def __init__(self) -> None:
        self.train_calls: list[dict[str, Any]] = []
        self.predict_calls: list[
            tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, float], dict[int, float], EngineConfig]
        ] = []

    def predict(
        self,
        bat_rows: list[dict[str, Any]],
        pitch_rows: list[dict[str, Any]],
        bat_pt: dict[int, float],
        pitch_pt: dict[int, float],
        config: EngineConfig,
    ) -> list[dict[str, Any]]:
        self.predict_calls.append((bat_rows, pitch_rows, bat_pt, pitch_pt, config))
        return [{"player_id": 99, "season": config.projected_season, "player_type": "batter"}]

    def train(
        self,
        bat_train_rows: list[dict[str, Any]],
        bat_holdout_rows: list[dict[str, Any]],
        pitch_train_rows: list[dict[str, Any]],
        pitch_holdout_rows: list[dict[str, Any]],
        bat_feature_cols: list[str],
        pitch_feature_cols: list[str],
        model_params: dict[str, Any],
        artifact_path: Path,
    ) -> dict[str, float]:
        self.train_calls.append(
            {
                "bat_train_rows": bat_train_rows,
                "bat_holdout_rows": bat_holdout_rows,
                "pitch_train_rows": pitch_train_rows,
                "pitch_holdout_rows": pitch_holdout_rows,
                "bat_feature_cols": bat_feature_cols,
                "pitch_feature_cols": pitch_feature_cols,
                "model_params": model_params,
                "artifact_path": artifact_path,
            }
        )
        return {"batter_rmse_avg": 0.01}


@pytest.mark.slow
class TestCompositeModelTrain:
    @pytest.fixture
    def model_params(self) -> dict[str, Any]:
        return {"batting_categories": ["hr"], "pitching_categories": ["so"], "engine": "gbm"}

    @pytest.fixture
    def assembler(self) -> SeasonAwareFakeAssembler:
        bat_2022 = [_make_batter_row(i, 2022) for i in range(1, 21)]
        bat_2023 = [_make_batter_row(i, 2023) for i in range(1, 11)]
        pitch_2022 = [_make_pitcher_row(i, 2022) for i in range(100, 120)]
        pitch_2023 = [_make_pitcher_row(i, 2023) for i in range(100, 110)]
        return SeasonAwareFakeAssembler(
            rows_by_season={2022: bat_2022, 2023: bat_2023},
            pitcher_rows_by_season={2022: pitch_2022, 2023: pitch_2023},
        )

    def test_train_returns_train_result(
        self,
        assembler: SeasonAwareFakeAssembler,
        model_params: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        model = CompositeModel(assembler=assembler, engine=GBMEngine(), group_lookup=_test_lookup)
        config = ModelConfig(seasons=[2022, 2023], model_params=model_params, artifacts_dir=str(tmp_path))
        result = model.train(config)
        assert isinstance(result, TrainResult)
        assert result.model_name == "composite"
        assert len(result.metrics) > 0

    def test_train_saves_artifacts(
        self,
        assembler: SeasonAwareFakeAssembler,
        model_params: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        model = CompositeModel(assembler=assembler, engine=GBMEngine(), group_lookup=_test_lookup)
        config = ModelConfig(seasons=[2022, 2023], model_params=model_params, artifacts_dir=str(tmp_path))
        result = model.train(config)
        artifact_path = Path(result.artifacts_path)
        assert (artifact_path / "batter_models.joblib").exists()
        assert (artifact_path / "pitcher_models.joblib").exists()

    def test_train_requires_at_least_2_seasons(
        self,
        assembler: SeasonAwareFakeAssembler,
        model_params: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        model = CompositeModel(assembler=assembler, engine=GBMEngine(), group_lookup=_test_lookup)
        config = ModelConfig(seasons=[2022], model_params=model_params, artifacts_dir=str(tmp_path))
        with pytest.raises(ValueError, match="at least 2 seasons"):
            model.train(config)

    def test_train_unsupported_engine_raises(
        self,
        assembler: SeasonAwareFakeAssembler,
        model_params: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        model = CompositeModel(assembler=assembler, engine=MarcelEngine(), group_lookup=_test_lookup)
        config = ModelConfig(seasons=[2022, 2023], model_params=model_params, artifacts_dir=str(tmp_path))
        with pytest.raises(ValueError, match="does not support train"):
            model.train(config)

    def test_train_passes_feature_columns(
        self,
        assembler: SeasonAwareFakeAssembler,
        model_params: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        engine = RecordingGBMEngine()
        model = CompositeModel(assembler=assembler, engine=engine, group_lookup=_test_lookup)
        config = ModelConfig(seasons=[2022, 2023], model_params=model_params, artifacts_dir=str(tmp_path))
        marcel_config = _build_marcel_config(config.model_params)
        batting_fs, pitching_fs = model._build_feature_sets(marcel_config, config)
        expected_bat_cols = feature_columns(batting_fs)
        expected_pitch_cols = feature_columns(pitching_fs)

        model.train(config)

        assert len(engine.train_calls) == 1
        call = engine.train_calls[0]
        assert call["bat_feature_cols"] == expected_bat_cols
        assert call["pitch_feature_cols"] == expected_pitch_cols


class TestCompositeAblate:
    def test_ablate_requires_at_least_2_seasons(self) -> None:
        assembler = SeasonAwareFakeAssembler()
        model = CompositeModel(assembler=assembler, engine=GBMEngine(), group_lookup=_test_lookup)
        config = ModelConfig(
            seasons=[2022],
            model_params={"batting_categories": ["hr"], "pitching_categories": ["so"]},
        )
        with pytest.raises(ValueError, match="at least 2 seasons"):
            model.ablate(config)

    def test_ablate_requires_gbm_engine(self) -> None:
        assembler = SeasonAwareFakeAssembler()
        model = CompositeModel(assembler=assembler, engine=MarcelEngine(), group_lookup=_test_lookup)
        config = ModelConfig(
            seasons=[2022, 2023],
            model_params={"batting_categories": ["hr"], "pitching_categories": ["so"]},
        )
        with pytest.raises(ValueError, match="does not support ablate"):
            model.ablate(config)


@pytest.fixture(scope="class")
def composite_ablation_result() -> AblationResult:
    bat_2022 = [_make_batter_row(i, 2022) for i in range(1, 31)]
    bat_2023 = [_make_batter_row(i, 2023) for i in range(1, 31)]
    pitch_2022 = [_make_pitcher_row(i, 2022) for i in range(100, 130)]
    pitch_2023 = [_make_pitcher_row(i, 2023) for i in range(100, 130)]
    assembler = SeasonAwareFakeAssembler(
        rows_by_season={2022: bat_2022, 2023: bat_2023},
        pitcher_rows_by_season={2022: pitch_2022, 2023: pitch_2023},
    )
    model = CompositeModel(assembler=assembler, engine=GBMEngine(), group_lookup=_test_lookup)
    config = ModelConfig(
        seasons=[2022, 2023],
        model_params={
            "batting_categories": ["hr"],
            "pitching_categories": ["so"],
            "n_repeats": 5,
        },
    )
    return model.ablate(config)


@pytest.mark.slow
class TestCompositeAblateResults:
    def test_returns_ablation_result(self, composite_ablation_result: AblationResult) -> None:
        assert isinstance(composite_ablation_result, AblationResult)
        assert composite_ablation_result.model_name == "composite"

    def test_returns_nonempty_impacts(self, composite_ablation_result: AblationResult) -> None:
        assert len(composite_ablation_result.feature_impacts) > 0

    def test_impacts_include_batter_features(self, composite_ablation_result: AblationResult) -> None:
        batter_keys = [k for k in composite_ablation_result.feature_impacts if k.startswith("batter:")]
        assert len(batter_keys) > 0

    def test_impacts_include_pitcher_features(self, composite_ablation_result: AblationResult) -> None:
        pitcher_keys = [k for k in composite_ablation_result.feature_impacts if k.startswith("pitcher:")]
        assert len(pitcher_keys) > 0

    def test_returns_standard_errors(self, composite_ablation_result: AblationResult) -> None:
        assert len(composite_ablation_result.feature_standard_errors) > 0
        assert set(composite_ablation_result.feature_standard_errors.keys()) == set(
            composite_ablation_result.feature_impacts.keys()
        )

    def test_standard_errors_non_negative(self, composite_ablation_result: AblationResult) -> None:
        for v in composite_ablation_result.feature_standard_errors.values():
            assert v >= 0.0

    def test_group_impacts_is_dict(self, composite_ablation_result: AblationResult) -> None:
        assert isinstance(composite_ablation_result.group_impacts, dict)

    def test_default_no_validation(self, composite_ablation_result: AblationResult) -> None:
        assert composite_ablation_result.validation_results == {}


class TestGBMPredictConfig:
    def test_gbm_predict_passes_artifact_path_and_feature_cols(self, tmp_path: Path) -> None:
        batting_rows = [_make_batter_row(1, 2023)]
        pitching_rows = [_make_pitcher_row(100, 2023)]
        assembler = FakeAssembler(batting_rows, pitching_rows)
        engine = RecordingGBMEngine()
        model = CompositeModel(assembler=assembler, engine=engine, group_lookup=_test_lookup)
        model_params: dict[str, Any] = {"batting_categories": ["hr"], "pitching_categories": ["so"]}
        config = ModelConfig(
            seasons=[2023],
            model_params=model_params,
            artifacts_dir=str(tmp_path),
        )
        model.predict(config)

        assert len(engine.predict_calls) == 1
        _, _, _, _, engine_config = engine.predict_calls[0]

        # artifact_path should be set for GBMEngine
        expected_artifact_path = Path(str(tmp_path)) / "composite" / "latest"
        assert engine_config.artifact_path == expected_artifact_path

        # feature columns should be populated
        assert len(engine_config.bat_feature_cols) > 0
        assert len(engine_config.pitch_feature_cols) > 0
        assert "age" in engine_config.bat_feature_cols
        assert "age" in engine_config.pitch_feature_cols


@pytest.fixture(scope="class")
def composite_multi_holdout_result() -> AblationResult:
    bat_rows = {s: [_make_batter_row(i, s) for i in range(1, 31)] for s in (2021, 2022, 2023)}
    pitch_rows = {s: [_make_pitcher_row(i, s) for i in range(100, 130)] for s in (2021, 2022, 2023)}
    assembler = SeasonAwareFakeAssembler(
        rows_by_season=bat_rows,
        pitcher_rows_by_season=pitch_rows,
    )
    model = CompositeModel(assembler=assembler, engine=GBMEngine(), group_lookup=_test_lookup)
    config = ModelConfig(
        seasons=[2021, 2022, 2023],
        model_params={
            "batting_categories": ["hr"],
            "pitching_categories": ["so"],
            "multi_holdout": True,
            "n_repeats": 5,
        },
    )
    return model.ablate(config)


@pytest.mark.slow
class TestCompositeAblateMultiHoldout:
    def test_returns_ablation_result(self, composite_multi_holdout_result: AblationResult) -> None:
        assert isinstance(composite_multi_holdout_result, AblationResult)
        assert composite_multi_holdout_result.model_name == "composite"

    def test_has_batter_and_pitcher_impacts(self, composite_multi_holdout_result: AblationResult) -> None:
        batter_keys = [k for k in composite_multi_holdout_result.feature_impacts if k.startswith("batter:")]
        pitcher_keys = [k for k in composite_multi_holdout_result.feature_impacts if k.startswith("pitcher:")]
        assert len(batter_keys) > 0
        assert len(pitcher_keys) > 0

    def test_se_non_negative(self, composite_multi_holdout_result: AblationResult) -> None:
        for v in composite_multi_holdout_result.feature_standard_errors.values():
            assert v >= 0.0
