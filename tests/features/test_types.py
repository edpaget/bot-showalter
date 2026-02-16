import dataclasses

import pytest

import re

from fantasy_baseball_manager.features.types import (
    AnyFeature,
    DatasetHandle,
    DatasetSplits,
    DeltaFeature,
    DerivedTransformFeature,
    Feature,
    FeatureBuilder,
    FeatureSet,
    Source,
    SourceRef,
    SpineFilter,
    TransformFeature,
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

    def test_statcast_value(self) -> None:
        assert Source.STATCAST.value == "statcast"

    def test_il_stint_value(self) -> None:
        assert Source.IL_STINT.value == "il_stint"

    def test_all_members(self) -> None:
        assert set(Source) == {
            Source.BATTING,
            Source.PITCHING,
            Source.PLAYER,
            Source.PROJECTION,
            Source.STATCAST,
            Source.IL_STINT,
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

    def test_system_default_none(self) -> None:
        f = Feature(name="hr_1", source=Source.BATTING, column="hr")
        assert f.system is None

    def test_system_field(self) -> None:
        f = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer")
        assert f.system == "steamer"

    def test_version_default_none(self) -> None:
        f = Feature(name="hr_1", source=Source.BATTING, column="hr")
        assert f.version is None

    def test_version_field(self) -> None:
        f = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer", version="2023.1")
        assert f.version == "2023.1"

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

    def test_system_returns_self(self) -> None:
        builder = FeatureBuilder(source=Source.PROJECTION, column="hr")
        result = builder.system("steamer")
        assert result is builder

    def test_system_sets_system_on_feature(self) -> None:
        feature = FeatureBuilder(source=Source.PROJECTION, column="hr").system("steamer").alias("steamer_hr")
        assert feature.system == "steamer"

    def test_system_default_none_on_feature(self) -> None:
        feature = FeatureBuilder(source=Source.BATTING, column="hr").alias("hr_0")
        assert feature.system is None

    def test_version_method(self) -> None:
        builder = FeatureBuilder(source=Source.PROJECTION, column="hr")
        result = builder.version("2023.1")
        assert result is builder
        feature = builder.system("steamer").alias("steamer_hr")
        assert feature.version == "2023.1"

    def test_defaults_without_chaining(self) -> None:
        feature = FeatureBuilder(source=Source.PITCHING, column="so").alias("so_0")
        assert feature.lag == 0
        assert feature.window == 1
        assert feature.aggregate is None
        assert feature.denominator is None
        assert feature.computed is None
        assert feature.system is None

    def test_projection_without_system_raises(self) -> None:
        builder = FeatureBuilder(source=Source.PROJECTION, column="hr")
        with pytest.raises(ValueError, match="system"):
            builder.alias("proj_hr")

    def test_projection_with_system_succeeds(self) -> None:
        feature = FeatureBuilder(source=Source.PROJECTION, column="hr").system("steamer").alias("steamer_hr")
        assert feature.source == Source.PROJECTION
        assert feature.system == "steamer"

    def test_percentile_returns_self(self) -> None:
        builder = FeatureBuilder(source=Source.PROJECTION, column="hr")
        result = builder.percentile(90)
        assert result is builder

    def test_percentile_sets_distribution_column(self) -> None:
        feature = (
            FeatureBuilder(source=Source.PROJECTION, column="hr")
            .percentile(90)
            .system("steamer")
            .alias("steamer_hr_p90")
        )
        assert feature.distribution_column == "p90"

    def test_percentile_validates_source(self) -> None:
        builder = FeatureBuilder(source=Source.BATTING, column="hr")
        with pytest.raises(ValueError, match="PROJECTION"):
            builder.percentile(90)

    def test_percentile_validates_values(self) -> None:
        builder = FeatureBuilder(source=Source.PROJECTION, column="hr")
        with pytest.raises(ValueError, match="10, 25, 50, 75, 90"):
            builder.percentile(99)

    def test_std_returns_self(self) -> None:
        builder = FeatureBuilder(source=Source.PROJECTION, column="hr")
        result = builder.std()
        assert result is builder

    def test_std_sets_distribution_column(self) -> None:
        feature = FeatureBuilder(source=Source.PROJECTION, column="hr").std().system("steamer").alias("steamer_hr_std")
        assert feature.distribution_column == "std"

    def test_std_validates_source(self) -> None:
        builder = FeatureBuilder(source=Source.BATTING, column="hr")
        with pytest.raises(ValueError, match="PROJECTION"):
            builder.std()


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

    def test_positions_returns_feature_for_player(self) -> None:
        ref = SourceRef(Source.PLAYER)
        feature = ref.positions()
        assert isinstance(feature, Feature)
        assert feature.name == "position"
        assert feature.source == Source.PLAYER
        assert feature.computed == "positions"

    def test_positions_raises_for_non_player_source(self) -> None:
        ref = SourceRef(Source.BATTING)
        with pytest.raises(ValueError, match="positions.*player"):
            ref.positions()

    def test_source_attribute(self) -> None:
        ref = SourceRef(Source.BATTING)
        assert ref.source == Source.BATTING


class TestDeltaFeature:
    def test_construction(self) -> None:
        left = Feature(name="actual_hr", source=Source.BATTING, column="hr", lag=0)
        right = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer")
        delta = DeltaFeature(name="hr_error", left=left, right=right)
        assert delta.name == "hr_error"
        assert delta.left is left
        assert delta.right is right

    def test_frozen(self) -> None:
        left = Feature(name="actual_hr", source=Source.BATTING, column="hr")
        right = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer")
        delta = DeltaFeature(name="hr_error", left=left, right=right)
        with pytest.raises(dataclasses.FrozenInstanceError):
            delta.name = "changed"  # type: ignore[misc]

    def test_any_feature_type(self) -> None:
        f: AnyFeature = Feature(name="hr_1", source=Source.BATTING, column="hr", lag=1)
        assert isinstance(f, Feature)
        d: AnyFeature = DeltaFeature(
            name="hr_error",
            left=Feature(name="actual_hr", source=Source.BATTING, column="hr"),
            right=Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer"),
        )
        assert isinstance(d, DeltaFeature)


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

    def test_version_field_affects_hash(self) -> None:
        f1 = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer", version="2023.1")
        f2 = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer", version="2023.2")
        fs1 = FeatureSet(name="test", features=(f1,), seasons=(2022,))
        fs2 = FeatureSet(name="test", features=(f2,), seasons=(2022,))
        assert fs1.version != fs2.version

    def test_system_affects_version(self) -> None:
        f1 = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer")
        f2 = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="zips")
        fs1 = FeatureSet(name="test", features=(f1,), seasons=(2022,))
        fs2 = FeatureSet(name="test", features=(f2,), seasons=(2022,))
        assert fs1.version != fs2.version

    def test_with_delta_feature(self) -> None:
        left = Feature(name="actual_hr", source=Source.BATTING, column="hr", lag=0)
        right = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer")
        delta = DeltaFeature(name="hr_error", left=left, right=right)
        fs = FeatureSet(name="test", features=(left, delta), seasons=(2022,))
        assert isinstance(fs.version, str)
        assert len(fs.version) == 12

    def test_delta_affects_version(self) -> None:
        left = Feature(name="actual_hr", source=Source.BATTING, column="hr", lag=0)
        right_a = Feature(name="steamer_hr", source=Source.PROJECTION, column="hr", system="steamer")
        right_b = Feature(name="zips_hr", source=Source.PROJECTION, column="hr", system="zips")
        fs1 = FeatureSet(name="test", features=(DeltaFeature(name="err", left=left, right=right_a),), seasons=(2022,))
        fs2 = FeatureSet(name="test", features=(DeltaFeature(name="err", left=left, right=right_b),), seasons=(2022,))
        assert fs1.version != fs2.version


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


def _sample_transform(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"out_a": 1.0, "out_b": 2.0}


def _different_transform(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"out_a": 99.0, "out_b": 99.0}


def _sample_derived_transform(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"derived_out": 42.0}


def _different_derived_transform(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"derived_out": 99.0}


class TestTransformFeature:
    def test_construction(self) -> None:
        tf = TransformFeature(
            name="pitch_mix",
            source=Source.STATCAST,
            columns=("pitch_type", "release_speed"),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a", "out_b"),
        )
        assert tf.name == "pitch_mix"
        assert tf.source == Source.STATCAST
        assert tf.columns == ("pitch_type", "release_speed")
        assert tf.group_by == ("player_id", "season")
        assert tf.outputs == ("out_a", "out_b")
        assert tf.version is None

    def test_frozen(self) -> None:
        tf = TransformFeature(
            name="pitch_mix",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            tf.name = "changed"  # type: ignore[misc]

    def test_any_feature_type(self) -> None:
        tf: AnyFeature = TransformFeature(
            name="pitch_mix",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
        )
        assert isinstance(tf, TransformFeature)

    def test_content_hash_includes_transform_source(self) -> None:
        tf = TransformFeature(
            name="pitch_mix",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
        )
        fs = FeatureSet(name="test", features=(tf,), seasons=(2023,))
        # Version should incorporate transform source code
        assert isinstance(fs.version, str)
        assert len(fs.version) == 12

    def test_content_hash_changes_when_transform_changes(self) -> None:
        tf1 = TransformFeature(
            name="t",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
        )
        tf2 = TransformFeature(
            name="t",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_different_transform,
            outputs=("out_a",),
        )
        fs1 = FeatureSet(name="test", features=(tf1,), seasons=(2023,))
        fs2 = FeatureSet(name="test", features=(tf2,), seasons=(2023,))
        assert fs1.version != fs2.version

    def test_explicit_version_overrides_source(self) -> None:
        tf1 = TransformFeature(
            name="t",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
            version="v1",
        )
        tf2 = TransformFeature(
            name="t",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_different_transform,
            outputs=("out_a",),
            version="v1",
        )
        fs1 = FeatureSet(name="test", features=(tf1,), seasons=(2023,))
        fs2 = FeatureSet(name="test", features=(tf2,), seasons=(2023,))
        # Same explicit version → same hash even though transforms differ
        assert fs1.version == fs2.version

    def test_different_explicit_versions_differ(self) -> None:
        tf1 = TransformFeature(
            name="t",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
            version="v1",
        )
        tf2 = TransformFeature(
            name="t",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
            version="v2",
        )
        fs1 = FeatureSet(name="test", features=(tf1,), seasons=(2023,))
        fs2 = FeatureSet(name="test", features=(tf2,), seasons=(2023,))
        assert fs1.version != fs2.version

    def test_lag_defaults_to_zero(self) -> None:
        tf = TransformFeature(
            name="pitch_mix",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
        )
        assert tf.lag == 0

    def test_lag_field(self) -> None:
        tf = TransformFeature(
            name="pitch_mix",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
            lag=1,
        )
        assert tf.lag == 1

    def test_with_lag_returns_new_instance(self) -> None:
        tf = TransformFeature(
            name="pitch_mix",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
        )
        lagged = tf.with_lag(1)
        assert lagged is not tf
        assert lagged.lag == 1
        assert tf.lag == 0  # original unchanged

    def test_with_lag_preserves_other_fields(self) -> None:
        tf = TransformFeature(
            name="pitch_mix",
            source=Source.STATCAST,
            columns=("pitch_type", "release_speed"),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a", "out_b"),
            version="v1",
        )
        lagged = tf.with_lag(2)
        assert lagged.name == tf.name
        assert lagged.source == tf.source
        assert lagged.columns == tf.columns
        assert lagged.group_by == tf.group_by
        assert lagged.transform is tf.transform
        assert lagged.outputs == tf.outputs
        assert lagged.version == tf.version
        assert lagged.lag == 2

    def test_lag_affects_feature_set_version(self) -> None:
        tf0 = TransformFeature(
            name="t",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
            lag=0,
        )
        tf1 = TransformFeature(
            name="t",
            source=Source.STATCAST,
            columns=("pitch_type",),
            group_by=("player_id", "season"),
            transform=_sample_transform,
            outputs=("out_a",),
            lag=1,
        )
        fs0 = FeatureSet(name="test", features=(tf0,), seasons=(2023,))
        fs1 = FeatureSet(name="test", features=(tf1,), seasons=(2023,))
        assert fs0.version != fs1.version


class TestDerivedTransformFeature:
    def test_construction(self) -> None:
        dtf = DerivedTransformFeature(
            name="weighted_avg",
            inputs=("hr_1", "hr_2", "hr_3"),
            group_by=("player_id", "season"),
            transform=_sample_derived_transform,
            outputs=("derived_out",),
        )
        assert dtf.name == "weighted_avg"
        assert dtf.inputs == ("hr_1", "hr_2", "hr_3")
        assert dtf.group_by == ("player_id", "season")
        assert dtf.outputs == ("derived_out",)
        assert dtf.version is None

    def test_frozen(self) -> None:
        dtf = DerivedTransformFeature(
            name="weighted_avg",
            inputs=("hr_1",),
            group_by=("player_id", "season"),
            transform=_sample_derived_transform,
            outputs=("derived_out",),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            dtf.name = "changed"  # type: ignore[misc]

    def test_any_feature_type(self) -> None:
        dtf: AnyFeature = DerivedTransformFeature(
            name="weighted_avg",
            inputs=("hr_1",),
            group_by=("player_id", "season"),
            transform=_sample_derived_transform,
            outputs=("derived_out",),
        )
        assert isinstance(dtf, DerivedTransformFeature)

    def test_content_hash_includes_derived_transform(self) -> None:
        dtf = DerivedTransformFeature(
            name="weighted_avg",
            inputs=("hr_1",),
            group_by=("player_id", "season"),
            transform=_sample_derived_transform,
            outputs=("derived_out",),
        )
        fs = FeatureSet(name="test", features=(dtf,), seasons=(2023,))
        assert isinstance(fs.version, str)
        assert len(fs.version) == 12

    def test_content_hash_changes_when_transform_changes(self) -> None:
        dtf1 = DerivedTransformFeature(
            name="t",
            inputs=("hr_1",),
            group_by=("player_id", "season"),
            transform=_sample_derived_transform,
            outputs=("derived_out",),
        )
        dtf2 = DerivedTransformFeature(
            name="t",
            inputs=("hr_1",),
            group_by=("player_id", "season"),
            transform=_different_derived_transform,
            outputs=("derived_out",),
        )
        fs1 = FeatureSet(name="test", features=(dtf1,), seasons=(2023,))
        fs2 = FeatureSet(name="test", features=(dtf2,), seasons=(2023,))
        assert fs1.version != fs2.version

    def test_explicit_version_overrides_source(self) -> None:
        dtf1 = DerivedTransformFeature(
            name="t",
            inputs=("hr_1",),
            group_by=("player_id", "season"),
            transform=_sample_derived_transform,
            outputs=("derived_out",),
            version="v1",
        )
        dtf2 = DerivedTransformFeature(
            name="t",
            inputs=("hr_1",),
            group_by=("player_id", "season"),
            transform=_different_derived_transform,
            outputs=("derived_out",),
            version="v1",
        )
        fs1 = FeatureSet(name="test", features=(dtf1,), seasons=(2023,))
        fs2 = FeatureSet(name="test", features=(dtf2,), seasons=(2023,))
        # Same explicit version → same hash even though transforms differ
        assert fs1.version == fs2.version
