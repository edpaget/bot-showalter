from fantasy_baseball_manager.features.types import Feature, FeatureSet, SpineFilter, TransformFeature
from fantasy_baseball_manager.models.statcast_gbm.features import (
    batter_feature_columns,
    batter_preseason_feature_columns,
    build_batter_feature_set,
    build_batter_preseason_set,
    build_batter_preseason_training_set,
    build_batter_training_set,
    build_pitcher_feature_set,
    build_pitcher_preseason_set,
    build_pitcher_preseason_training_set,
    build_pitcher_training_set,
    pitcher_feature_columns,
    pitcher_preseason_feature_columns,
)


class TestBatterFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_batter_feature_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_has_source_filter(self) -> None:
        fs = build_batter_feature_set([2023])
        assert fs.source_filter == "fangraphs"

    def test_has_batter_spine_filter(self) -> None:
        fs = build_batter_feature_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="batter")

    def test_includes_age(self) -> None:
        fs = build_batter_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_includes_statcast_transforms(self) -> None:
        fs = build_batter_feature_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "batted_ball" in transform_names
        assert "plate_discipline" in transform_names
        assert "expected_stats" in transform_names

    def test_includes_batting_lags(self) -> None:
        fs = build_batter_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "pa_1" in names
        assert "pa_2" in names
        assert "hr_1" in names
        assert "hr_2" in names


class TestBatterTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_batter_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_avg" in names
        assert "target_obp" in names
        assert "target_slg" in names
        assert "target_woba" in names
        assert "target_h" in names
        assert "target_hr" in names

    def test_includes_input_features(self) -> None:
        fs = build_batter_training_set([2023])
        names: list[str] = []
        for f in fs.features:
            if isinstance(f, TransformFeature):
                names.append(f.name)
            elif isinstance(f, Feature):
                names.append(f.name)
        assert "age" in names
        assert "pa_1" in names
        assert "batted_ball" in names

    def test_name(self) -> None:
        fs = build_batter_training_set([2023])
        assert fs.name == "statcast_gbm_batting_train"


class TestBatterFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = batter_feature_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_no_target_columns(self) -> None:
        columns = batter_feature_columns()
        assert not any(c.startswith("target_") for c in columns)

    def test_includes_feature_names(self) -> None:
        columns = batter_feature_columns()
        assert "age" in columns
        assert "pa_1" in columns
        # TransformFeature outputs should be expanded
        assert "avg_exit_velo" in columns
        assert "chase_rate" in columns
        assert "xba" in columns


class TestPitcherFeatureSet:
    def test_returns_feature_set(self) -> None:
        fs = build_pitcher_feature_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_has_source_filter(self) -> None:
        fs = build_pitcher_feature_set([2023])
        assert fs.source_filter == "fangraphs"

    def test_has_pitcher_spine_filter(self) -> None:
        fs = build_pitcher_feature_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="pitcher")

    def test_includes_age(self) -> None:
        fs = build_pitcher_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_includes_statcast_transforms(self) -> None:
        fs = build_pitcher_feature_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "pitch_mix" in transform_names
        assert "spin_profile" in transform_names
        assert "plate_discipline" in transform_names

    def test_includes_pitching_lags(self) -> None:
        fs = build_pitcher_feature_set([2023])
        names = [f.name for f in fs.features]
        assert "ip_1" in names
        assert "ip_2" in names
        assert "so_1" in names
        assert "so_2" in names


class TestPitcherTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_pitcher_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_era" in names
        assert "target_fip" in names
        assert "target_k_per_9" in names
        assert "target_bb_per_9" in names
        assert "target_whip" in names
        # Counting stats for derived targets
        assert "target_h" in names
        assert "target_hr" in names
        assert "target_ip" in names
        assert "target_so" in names

    def test_includes_input_features(self) -> None:
        fs = build_pitcher_training_set([2023])
        names: list[str] = []
        for f in fs.features:
            if isinstance(f, TransformFeature):
                names.append(f.name)
            elif isinstance(f, Feature):
                names.append(f.name)
        assert "age" in names
        assert "ip_1" in names
        assert "pitch_mix" in names

    def test_name(self) -> None:
        fs = build_pitcher_training_set([2023])
        assert fs.name == "statcast_gbm_pitching_train"


class TestPitcherFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = pitcher_feature_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_no_target_columns(self) -> None:
        columns = pitcher_feature_columns()
        assert not any(c.startswith("target_") for c in columns)

    def test_includes_feature_names(self) -> None:
        columns = pitcher_feature_columns()
        assert "age" in columns
        assert "ip_1" in columns
        # TransformFeature outputs should be expanded
        assert "avg_spin_rate" in columns
        assert "ff_pct" in columns
        assert "chase_rate" in columns


class TestBatterPreseasonSet:
    def test_returns_feature_set(self) -> None:
        fs = build_batter_preseason_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_name(self) -> None:
        fs = build_batter_preseason_set([2023])
        assert fs.name == "statcast_gbm_batting_preseason"

    def test_has_batter_spine_filter(self) -> None:
        fs = build_batter_preseason_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="batter")

    def test_includes_age(self) -> None:
        fs = build_batter_preseason_set([2023])
        names = [f.name for f in fs.features]
        assert "age" in names

    def test_includes_batting_lags(self) -> None:
        fs = build_batter_preseason_set([2023])
        names = [f.name for f in fs.features]
        assert "pa_1" in names
        assert "pa_2" in names

    def test_statcast_transforms_are_lagged(self) -> None:
        fs = build_batter_preseason_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        assert len(transforms) > 0
        for tf in transforms:
            assert tf.lag == 1, f"{tf.name} should have lag=1 but has lag={tf.lag}"

    def test_statcast_transform_names(self) -> None:
        fs = build_batter_preseason_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "batted_ball" in transform_names
        assert "plate_discipline" in transform_names
        assert "expected_stats" in transform_names

    def test_different_version_from_true_talent(self) -> None:
        tt = build_batter_feature_set([2023])
        ps = build_batter_preseason_set([2023])
        assert tt.version != ps.version


class TestBatterPreseasonTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_batter_preseason_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_avg" in names
        assert "target_obp" in names
        assert "target_slg" in names

    def test_name(self) -> None:
        fs = build_batter_preseason_training_set([2023])
        assert fs.name == "statcast_gbm_batting_preseason_train"

    def test_statcast_transforms_are_lagged(self) -> None:
        fs = build_batter_preseason_training_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        for tf in transforms:
            assert tf.lag == 1


class TestBatterPreseasonFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = batter_preseason_feature_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_no_target_columns(self) -> None:
        columns = batter_preseason_feature_columns()
        assert not any(c.startswith("target_") for c in columns)

    def test_same_output_columns_as_true_talent(self) -> None:
        """Preseason uses same output names (same transform outputs)."""
        tt_cols = batter_feature_columns()
        ps_cols = batter_preseason_feature_columns()
        assert tt_cols == ps_cols


class TestPitcherPreseasonSet:
    def test_returns_feature_set(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        assert isinstance(fs, FeatureSet)

    def test_name(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        assert fs.name == "statcast_gbm_pitching_preseason"

    def test_has_pitcher_spine_filter(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        assert fs.spine_filter == SpineFilter(player_type="pitcher")

    def test_statcast_transforms_are_lagged(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        assert len(transforms) > 0
        for tf in transforms:
            assert tf.lag == 1

    def test_statcast_transform_names(self) -> None:
        fs = build_pitcher_preseason_set([2023])
        transform_names = [f.name for f in fs.features if isinstance(f, TransformFeature)]
        assert "pitch_mix" in transform_names
        assert "spin_profile" in transform_names
        assert "plate_discipline" in transform_names


class TestPitcherPreseasonTrainingSet:
    def test_includes_targets(self) -> None:
        fs = build_pitcher_preseason_training_set([2023])
        names = [f.name for f in fs.features if isinstance(f, Feature)]
        assert "target_era" in names
        assert "target_fip" in names

    def test_name(self) -> None:
        fs = build_pitcher_preseason_training_set([2023])
        assert fs.name == "statcast_gbm_pitching_preseason_train"

    def test_statcast_transforms_are_lagged(self) -> None:
        fs = build_pitcher_preseason_training_set([2023])
        transforms = [f for f in fs.features if isinstance(f, TransformFeature)]
        for tf in transforms:
            assert tf.lag == 1


class TestPitcherPreseasonFeatureColumns:
    def test_returns_list_of_strings(self) -> None:
        columns = pitcher_preseason_feature_columns()
        assert isinstance(columns, list)
        assert all(isinstance(c, str) for c in columns)

    def test_no_target_columns(self) -> None:
        columns = pitcher_preseason_feature_columns()
        assert not any(c.startswith("target_") for c in columns)

    def test_same_output_columns_as_true_talent(self) -> None:
        tt_cols = pitcher_feature_columns()
        ps_cols = pitcher_preseason_feature_columns()
        assert tt_cols == ps_cols
