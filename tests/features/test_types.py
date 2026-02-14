import dataclasses

import pytest

import re

from fantasy_baseball_manager.features.types import (
    DatasetHandle,
    DatasetSplits,
    Feature,
    FeatureBuilder,
    FeatureSet,
    Source,
    SourceRef,
    SpineFilter,
)


class TestSource:
    def test_batting_value(self) -> None:
        assert Source.BATTING.value == "batting"

    def test_pitching_value(self) -> None:
        assert Source.PITCHING.value == "pitching"

    def test_player_value(self) -> None:
        assert Source.PLAYER.value == "player"

    def test_projection_value(self) -> None:
        assert Source.PROJECTION.value == "projection"

    def test_all_members(self) -> None:
        assert set(Source) == {
            Source.BATTING,
            Source.PITCHING,
            Source.PLAYER,
            Source.PROJECTION,
        }


class TestSpineFilter:
    def test_defaults_all_none(self) -> None:
        sf = SpineFilter()
        assert sf.min_pa is None
        assert sf.min_ip is None
        assert sf.player_type is None

    def test_with_values(self) -> None:
        sf = SpineFilter(min_pa=50, min_ip=10.0, player_type="batter")
        assert sf.min_pa == 50
        assert sf.min_ip == 10.0
        assert sf.player_type == "batter"

    def test_frozen(self) -> None:
        sf = SpineFilter()
        with pytest.raises(dataclasses.FrozenInstanceError):
            sf.min_pa = 100  # type: ignore[misc]


class TestFeature:
    def test_required_fields(self) -> None:
        f = Feature(name="hr_1", source=Source.BATTING, column="hr")
        assert f.name == "hr_1"
        assert f.source == Source.BATTING
        assert f.column == "hr"

    def test_defaults(self) -> None:
        f = Feature(name="hr_1", source=Source.BATTING, column="hr")
        assert f.lag == 0
        assert f.window == 1
        assert f.aggregate is None
        assert f.denominator is None
        assert f.computed is None

    def test_all_fields(self) -> None:
        f = Feature(
            name="hr_rate_3yr",
            source=Source.BATTING,
            column="hr",
            lag=1,
            window=3,
            aggregate="mean",
            denominator="pa",
        )
        assert f.lag == 1
        assert f.window == 3
        assert f.aggregate == "mean"
        assert f.denominator == "pa"

    def test_computed_feature(self) -> None:
        f = Feature(name="age", source=Source.PLAYER, column="", computed="age")
        assert f.computed == "age"

    def test_frozen(self) -> None:
        f = Feature(name="hr_1", source=Source.BATTING, column="hr")
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.lag = 5  # type: ignore[misc]


class TestFeatureBuilder:
    def test_lag_returns_self(self) -> None:
        builder = FeatureBuilder(source=Source.BATTING, column="hr")
        result = builder.lag(1)
        assert result is builder

    def test_rolling_mean_returns_self(self) -> None:
        builder = FeatureBuilder(source=Source.BATTING, column="hr")
        result = builder.rolling_mean(3)
        assert result is builder

    def test_rolling_sum_returns_self(self) -> None:
        builder = FeatureBuilder(source=Source.BATTING, column="hr")
        result = builder.rolling_sum(3)
        assert result is builder

    def test_per_returns_self(self) -> None:
        builder = FeatureBuilder(source=Source.BATTING, column="hr")
        result = builder.per("pa")
        assert result is builder

    def test_alias_returns_feature(self) -> None:
        builder = FeatureBuilder(source=Source.BATTING, column="hr")
        feature = builder.lag(1).alias("hr_1")
        assert isinstance(feature, Feature)
        assert feature.name == "hr_1"
        assert feature.source == Source.BATTING
        assert feature.column == "hr"
        assert feature.lag == 1

    def test_alias_returns_frozen_feature(self) -> None:
        feature = FeatureBuilder(source=Source.BATTING, column="hr").alias("hr_0")
        with pytest.raises(dataclasses.FrozenInstanceError):
            feature.lag = 5  # type: ignore[misc]

    def test_rolling_mean_sets_window_and_aggregate(self) -> None:
        feature = FeatureBuilder(source=Source.BATTING, column="hr").rolling_mean(3).alias("hr_3yr")
        assert feature.window == 3
        assert feature.aggregate == "mean"

    def test_rolling_sum_sets_window_and_aggregate(self) -> None:
        feature = FeatureBuilder(source=Source.BATTING, column="hr").rolling_sum(5).alias("hr_5yr")
        assert feature.window == 5
        assert feature.aggregate == "sum"

    def test_per_sets_denominator(self) -> None:
        feature = FeatureBuilder(source=Source.BATTING, column="hr").per("pa").alias("hr_rate")
        assert feature.denominator == "pa"

    def test_complex_chain(self) -> None:
        feature = (
            FeatureBuilder(source=Source.BATTING, column="hr").lag(1).per("pa").rolling_mean(3).alias("hr_rate_3yr")
        )
        assert feature.name == "hr_rate_3yr"
        assert feature.source == Source.BATTING
        assert feature.column == "hr"
        assert feature.lag == 1
        assert feature.denominator == "pa"
        assert feature.window == 3
        assert feature.aggregate == "mean"

    def test_defaults_without_chaining(self) -> None:
        feature = FeatureBuilder(source=Source.PITCHING, column="so").alias("so_0")
        assert feature.lag == 0
        assert feature.window == 1
        assert feature.aggregate is None
        assert feature.denominator is None
        assert feature.computed is None


class TestSourceRef:
    def test_col_returns_feature_builder(self) -> None:
        ref = SourceRef(Source.BATTING)
        builder = ref.col("hr")
        assert isinstance(builder, FeatureBuilder)

    def test_col_binds_correct_source(self) -> None:
        ref = SourceRef(Source.PITCHING)
        feature = ref.col("so").alias("so_0")
        assert feature.source == Source.PITCHING
        assert feature.column == "so"

    def test_col_creates_fresh_builder_each_call(self) -> None:
        ref = SourceRef(Source.BATTING)
        b1 = ref.col("hr")
        b2 = ref.col("hr")
        assert b1 is not b2

    def test_age_returns_feature_for_player(self) -> None:
        ref = SourceRef(Source.PLAYER)
        feature = ref.age()
        assert isinstance(feature, Feature)
        assert feature.name == "age"
        assert feature.source == Source.PLAYER
        assert feature.computed == "age"

    def test_age_raises_for_non_player_source(self) -> None:
        ref = SourceRef(Source.BATTING)
        with pytest.raises(ValueError, match="age.*player"):
            ref.age()

    def test_age_raises_for_pitching(self) -> None:
        ref = SourceRef(Source.PITCHING)
        with pytest.raises(ValueError, match="age.*player"):
            ref.age()

    def test_source_attribute(self) -> None:
        ref = SourceRef(Source.BATTING)
        assert ref.source == Source.BATTING


class TestFeatureSet:
    def _make_features(self) -> tuple[Feature, ...]:
        return (
            Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1),
            Feature(name="pa_1", source=Source.BATTING, column="pa", lag=1),
        )

    def test_construction(self) -> None:
        features = self._make_features()
        fs = FeatureSet(
            name="test",
            features=features,
            seasons=(2022, 2023),
        )
        assert fs.name == "test"
        assert fs.features == features
        assert fs.seasons == (2022, 2023)
        assert fs.source_filter is None
        assert fs.spine_filter == SpineFilter()

    def test_version_is_auto_computed(self) -> None:
        fs = FeatureSet(
            name="test",
            features=self._make_features(),
            seasons=(2022, 2023),
        )
        assert isinstance(fs.version, str)
        assert len(fs.version) == 12

    def test_version_is_hex(self) -> None:
        fs = FeatureSet(
            name="test",
            features=self._make_features(),
            seasons=(2022, 2023),
        )
        assert re.fullmatch(r"[0-9a-f]{12}", fs.version)

    def test_identical_inputs_same_version(self) -> None:
        kwargs: dict[str, object] = dict(
            name="test",
            features=self._make_features(),
            seasons=(2022, 2023),
            source_filter="fangraphs",
            spine_filter=SpineFilter(min_pa=50),
        )
        fs1 = FeatureSet(**kwargs)  # type: ignore[arg-type]
        fs2 = FeatureSet(**kwargs)  # type: ignore[arg-type]
        assert fs1.version == fs2.version

    def test_different_features_different_version(self) -> None:
        fs1 = FeatureSet(
            name="test",
            features=self._make_features(),
            seasons=(2022,),
        )
        fs2 = FeatureSet(
            name="test",
            features=(Feature(name="so_1", source=Source.PITCHING, column="so", lag=1),),
            seasons=(2022,),
        )
        assert fs1.version != fs2.version

    def test_different_seasons_different_version(self) -> None:
        features = self._make_features()
        fs1 = FeatureSet(name="test", features=features, seasons=(2022,))
        fs2 = FeatureSet(name="test", features=features, seasons=(2023,))
        assert fs1.version != fs2.version

    def test_different_source_filter_different_version(self) -> None:
        features = self._make_features()
        fs1 = FeatureSet(name="test", features=features, seasons=(2022,), source_filter="fangraphs")
        fs2 = FeatureSet(name="test", features=features, seasons=(2022,), source_filter="bbref")
        assert fs1.version != fs2.version

    def test_different_spine_filter_different_version(self) -> None:
        features = self._make_features()
        fs1 = FeatureSet(name="test", features=features, seasons=(2022,), spine_filter=SpineFilter(min_pa=50))
        fs2 = FeatureSet(name="test", features=features, seasons=(2022,), spine_filter=SpineFilter(min_pa=100))
        assert fs1.version != fs2.version

    def test_frozen(self) -> None:
        fs = FeatureSet(name="test", features=self._make_features(), seasons=(2022,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            fs.name = "changed"  # type: ignore[misc]

    def test_name_does_not_affect_version(self) -> None:
        features = self._make_features()
        fs1 = FeatureSet(name="alpha", features=features, seasons=(2022,))
        fs2 = FeatureSet(name="beta", features=features, seasons=(2022,))
        assert fs1.version == fs2.version


class TestDatasetHandle:
    def test_construction(self) -> None:
        h = DatasetHandle(
            dataset_id=1,
            feature_set_id=42,
            table_name="ds_42",
            row_count=500,
            seasons=(2022, 2023),
        )
        assert h.dataset_id == 1
        assert h.feature_set_id == 42
        assert h.table_name == "ds_42"
        assert h.row_count == 500
        assert h.seasons == (2022, 2023)

    def test_frozen(self) -> None:
        h = DatasetHandle(
            dataset_id=1,
            feature_set_id=42,
            table_name="ds_42",
            row_count=500,
            seasons=(2022,),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            h.row_count = 999  # type: ignore[misc]


class TestDatasetSplits:
    def _make_handle(self, dataset_id: int, table_name: str) -> DatasetHandle:
        return DatasetHandle(
            dataset_id=dataset_id,
            feature_set_id=42,
            table_name=table_name,
            row_count=100,
            seasons=(2022,),
        )

    def test_construction(self) -> None:
        train = self._make_handle(1, "ds_42_train")
        val = self._make_handle(2, "ds_42_val")
        holdout = self._make_handle(3, "ds_42_holdout")
        splits = DatasetSplits(train=train, validation=val, holdout=holdout)
        assert splits.train is train
        assert splits.validation is val
        assert splits.holdout is holdout

    def test_optional_splits(self) -> None:
        train = self._make_handle(1, "ds_42_train")
        splits = DatasetSplits(train=train, validation=None, holdout=None)
        assert splits.validation is None
        assert splits.holdout is None

    def test_frozen(self) -> None:
        train = self._make_handle(1, "ds_42_train")
        splits = DatasetSplits(train=train, validation=None, holdout=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            splits.train = train  # type: ignore[misc]
