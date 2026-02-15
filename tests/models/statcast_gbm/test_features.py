from fantasy_baseball_manager.features.types import Feature, FeatureSet, SpineFilter, TransformFeature
from fantasy_baseball_manager.models.statcast_gbm.features import (
    batter_feature_columns,
    build_batter_feature_set,
    build_batter_training_set,
    build_pitcher_feature_set,
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
