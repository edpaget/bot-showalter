import importlib
from typing import Any

import pytest

import fantasy_baseball_manager.features.group_library
from fantasy_baseball_manager.features.groups import list_groups
from fantasy_baseball_manager.features.types import DatasetHandle, DatasetSplits, FeatureSet
import fantasy_baseball_manager.models.composite  # noqa: F401 — trigger alias registration
from fantasy_baseball_manager.models.composite.model import (
    DEFAULT_GROUPS,
    CompositeModel,
    _build_marcel_config,
    _resolve_group,
)
from fantasy_baseball_manager.models.composite.features import (
    build_composite_batting_features,
    build_composite_pitching_features,
)
from fantasy_baseball_manager.models.marcel.types import MarcelConfig
from fantasy_baseball_manager.models.registry import _clear, get, register, register_alias
from fantasy_baseball_manager.models.protocols import (
    Evaluable,
    FineTunable,
    Model,
    ModelConfig,
    Predictable,
    Preparable,
    Trainable,
)


@pytest.fixture(autouse=True)
def _ensure_groups_registered() -> None:
    """Re-register static feature groups if a prior test cleared the registry."""
    if not list_groups():
        importlib.reload(fantasy_baseball_manager.features.group_library)


class TestCompositeModelProtocol:
    def test_is_model(self) -> None:
        assert isinstance(CompositeModel(), Model)

    def test_is_preparable(self) -> None:
        assert isinstance(CompositeModel(), Preparable)

    def test_is_predictable(self) -> None:
        assert isinstance(CompositeModel(), Predictable)

    def test_is_not_trainable(self) -> None:
        assert not isinstance(CompositeModel(), Trainable)

    def test_is_not_evaluable(self) -> None:
        assert not isinstance(CompositeModel(), Evaluable)

    def test_is_not_finetuneable(self) -> None:
        assert not isinstance(CompositeModel(), FineTunable)

    def test_name(self) -> None:
        assert CompositeModel().name == "composite"

    def test_name_uses_model_name_param(self) -> None:
        assert CompositeModel(model_name="composite-mle").name == "composite-mle"

    def test_supported_operations(self) -> None:
        assert CompositeModel().supported_operations == frozenset({"prepare", "predict"})

    def test_artifact_type(self) -> None:
        assert CompositeModel().artifact_type == "none"


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


class TestResolveGroup:
    def test_resolve_group_static(self) -> None:
        config = MarcelConfig()
        group = _resolve_group("age", config)
        assert group.name == "age"
        assert group.player_type == "both"

    def test_resolve_group_batting_counting_lags(self) -> None:
        config = MarcelConfig(batting_categories=("hr", "rbi"), batting_weights=(5.0, 4.0))
        group = _resolve_group("batting_counting_lags", config)
        assert group.name == "batting_counting_lags"
        assert group.player_type == "batter"
        # 2 lags × (pa + hr + rbi) = 6 features
        assert len(group.features) == 6

    def test_resolve_group_pitching_counting_lags(self) -> None:
        config = MarcelConfig(pitching_categories=("so", "bb"), pitching_weights=(5.0, 4.0))
        group = _resolve_group("pitching_counting_lags", config)
        assert group.name == "pitching_counting_lags"
        assert group.player_type == "pitcher"
        # 2 lags × (ip + g + gs + so + bb) = 10 features
        assert len(group.features) == 10

    def test_resolve_group_unknown_raises(self) -> None:
        config = MarcelConfig()
        with pytest.raises(KeyError):
            _resolve_group("nonexistent_group_xyz", config)

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
        model = CompositeModel(assembler=assembler)
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
        model = CompositeModel(assembler=assembler)
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
        model = CompositeModel(assembler=assembler)
        config = ModelConfig(seasons=[2023], model_params={"batting_categories": ["hr"]})
        batting_fs, pitching_fs = model._build_feature_sets(_build_marcel_config(config.model_params), config)
        assert "batting" in batting_fs.name
        assert "pitching" in pitching_fs.name


class TestDefaultGroupsRegression:
    """Verify new group-based feature building produces the same features as the old inline builders."""

    def test_default_batting_features_match_inline(self) -> None:
        marcel_config = MarcelConfig(batting_categories=("hr", "rbi", "sb"))
        config = ModelConfig(seasons=[2023], model_params={"batting_categories": ["hr", "rbi", "sb"]})
        model = CompositeModel(assembler=FakeAssembler([]))
        batting_fs, _ = model._build_feature_sets(marcel_config, config)
        new_names = {f.name for f in batting_fs.features}

        old_features = build_composite_batting_features(marcel_config.batting_categories, marcel_config.batting_weights)
        old_names = {f.name for f in old_features}
        assert new_names == old_names

    def test_default_pitching_features_match_inline(self) -> None:
        marcel_config = MarcelConfig(pitching_categories=("so", "bb", "era"))
        config = ModelConfig(seasons=[2023], model_params={"pitching_categories": ["so", "bb", "era"]})
        model = CompositeModel(assembler=FakeAssembler([]))
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
        result = CompositeModel(assembler=assembler).predict(config)
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
        result = CompositeModel(assembler=assembler).predict(config)
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
        result = CompositeModel(assembler=assembler).predict(config)
        pitcher_preds = [p for p in result.predictions if p.get("player_type") == "pitcher"]
        assert len(pitcher_preds) == 1
        assert pitcher_preds[0]["ip"] == 150.0

    def test_predict_empty_data(self) -> None:
        assembler = FakeAssembler(batting_rows=[], pitching_rows=[])
        config = ModelConfig(
            seasons=[2023],
            model_params={"batting_categories": ["hr"]},
        )
        result = CompositeModel(assembler=assembler).predict(config)
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
        result = CompositeModel(assembler=assembler).predict(config)
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
        result = CompositeModel(assembler=assembler, model_name="composite-mle").predict(config)
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
        result = CompositeModel(assembler=assembler).predict(config)
        assert result.predictions[0]["season"] == 2024
